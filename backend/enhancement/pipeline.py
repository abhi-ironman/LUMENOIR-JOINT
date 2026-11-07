"""Enhancement pipeline orchestration.

The orchestrator runs the implemented stages in order and reports per-stage
timings, artifacts and failures. The stage classes delegate to the existing
LumenScan, LumenRetinex, Zero-DCE++, LumenFuse, EdgeGuard and HueLock modules;
no algorithm logic lives here.

The pipeline is stateless. Slow CPU work is executed by the API in a thread
pool; the Zero-DCE++ ModelManager serialises model inference internally.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass

from PIL import Image

from ..config import Settings
from ..inference.errors import DeviceUnavailableError, InferenceError
from ..inference.manager import ModelManager
from ..runtime.config import PipelineRuntime, resolve_runtime
from ..timing import Stopwatch
from .phase_engine import PhaseEngine, PhaseEngineConfig, PhaseOverrides
from .shared import SHARED_STAGES, CachedStage, SharedStageCache
from .stages import PipelineStage, StageContext, StageNotImplemented, default_stages
from .types import (
    ENHANCEMENT_MODES,
    MODE_LABELS,
    PLANNED_ARTIFACTS,
    Artifact,
    ArtifactKind,
    EnhancementMode,
    ModeCapability,
    PipelineResult,
    ProcessingStatus,
    StageCapability,
    StageName,
    StageReport,
    StageStatus,
)

logger = logging.getLogger(__name__)

# Which stages each selectable mode needs. The remaining stages are still
# reported as pending so the status endpoint never hides missing work.
REQUIRED_STAGES_BY_MODE: dict[EnhancementMode, tuple[StageName, ...]] = {
    "classical": (
        StageName.ANALYSIS,
        StageName.CLASSICAL_CV,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
    "deep_learning": (
        StageName.ANALYSIS,
        StageName.DEEP_LEARNING,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
    "hybrid": (
        StageName.ANALYSIS,
        StageName.CLASSICAL_CV,
        StageName.DEEP_LEARNING,
        StageName.FUSION,
        StageName.DENOISING,
        StageName.COLOUR_CORRECTION,
        StageName.QUALITY_ASSESSMENT,
    ),
}


@dataclass(frozen=True)
class PipelineCapabilities:
    """Static description of the pipeline for the status endpoint."""

    stages: tuple[StageCapability, ...]
    modes: tuple[ModeCapability, ...]

    @property
    def ready(self) -> bool:
        return any(mode.available for mode in self.modes)


class EnhancementPipeline:
    """Drives the ordered stages and reports what is missing."""

    def __init__(
        self,
        *,
        stages: Sequence[PipelineStage] | None = None,
        model_manager: ModelManager | None = None,
        settings: Settings | None = None,
        runtime: PipelineRuntime | None = None,
    ) -> None:
        self._stages: tuple[PipelineStage, ...] = tuple(
            stages if stages is not None else default_stages()
        )
        self.model_manager = model_manager
        self.settings = settings
        self._runtime = runtime
        self._ops = None
        self._runtime_lock = threading.Lock()
        self._execution_lock = threading.Lock()

        self._by_name = {stage.name: stage for stage in self._stages}
        if len(self._by_name) != len(self._stages):
            raise ValueError("Pipeline stages must be unique.")

    def _execution_ops(self):
        """Resolve the backend lazily so importing the API never imports torch."""
        try:
            if self._runtime is None:
                with self._runtime_lock:
                    if self._runtime is None:
                        self._runtime = resolve_runtime()
            if not self._runtime.is_gpu:
                return None
            if self._ops is None:
                with self._runtime_lock:
                    if self._ops is None:
                        from ..runtime.torch_ops import TorchImageOps

                        self._ops = TorchImageOps(self._runtime)
            return self._ops
        except Exception as exc:
            target = getattr(self._runtime, "device", None) or "accelerator"
            raise DeviceUnavailableError(target, str(exc)) from exc

    # -- introspection -----------------------------------------------------

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    def stages_for(self, mode: EnhancementMode) -> tuple[PipelineStage, ...]:
        """Return the stages required by ``mode`` in pipeline order."""
        required = set(REQUIRED_STAGES_BY_MODE[mode])
        return tuple(stage for stage in self._stages if stage.name in required)

    def capabilities(self) -> PipelineCapabilities:
        stage_caps = tuple(
            StageCapability(
                name=stage.name,
                status=stage.status,
                description=stage.description,
            )
            for stage in self._stages
        )

        model_ready = (
            True if self.model_manager is None else self.model_manager.status.ready
        )

        modes = tuple(
            ModeCapability(
                mode=mode,
                label=MODE_LABELS[mode],
                available=(
                    all(
                        stage.status is StageStatus.IMPLEMENTED
                        for stage in self.stages_for(mode)
                    )
                    and (
                        StageName.DEEP_LEARNING not in REQUIRED_STAGES_BY_MODE[mode]
                        or model_ready
                    )
                ),
                required_stages=REQUIRED_STAGES_BY_MODE[mode],
            )
            for mode in ENHANCEMENT_MODES
        )

        return PipelineCapabilities(stages=stage_caps, modes=modes)

    # -- execution ---------------------------------------------------------

    def run(
        self,
        image: Image.Image,
        *,
        mode: EnhancementMode = "hybrid",
        capture_diagnostics: bool = False,
        cache: SharedStageCache | None = None,
        closed_loop: bool | None = None,
    ) -> PipelineResult:
        """Run the selected mode, optionally using the bounded Phase Engine.

        Closed-loop runs are opt-in through closed_loop=True or
        LUMENOIR_CLOSED_LOOP_ENABLED=true. Every retry starts from the source
        image and bypasses the shared stage cache because its parameters differ.
        """
        if mode not in ENHANCEMENT_MODES:
            raise ValueError(f"Unsupported enhancement mode: {mode!r}.")

        ops = self._execution_ops()
        lock = self._execution_lock if ops is not None else nullcontext()
        with lock:
            enabled = closed_loop
            if enabled is None:
                enabled = os.getenv(
                    "LUMENOIR_CLOSED_LOOP_ENABLED", "false"
                ).strip().lower() in {"1", "true", "yes", "on"}

            if enabled:
                engine = PhaseEngine(PhaseEngineConfig())
                return engine.run(
                    original=image,
                    run_pass=lambda overrides: self._run_once(
                        image,
                        mode=mode,
                        capture_diagnostics=capture_diagnostics,
                        cache=None,
                        ops=ops,
                        phase_overrides=overrides,
                    ),
                )

            return self._run_once(
                image,
                mode=mode,
                capture_diagnostics=capture_diagnostics,
                cache=cache,
                ops=ops,
                phase_overrides=PhaseOverrides(),
            )

    def _run_once(
        self,
        image: Image.Image,
        *,
        mode: EnhancementMode,
        capture_diagnostics: bool,
        cache: SharedStageCache | None,
        ops,
        phase_overrides: PhaseOverrides,
    ) -> PipelineResult:
        context = StageContext(
            image=image,
            mode=mode,
            original_size=(image.width, image.height),
            model_manager=self.model_manager,
            capture_diagnostics=capture_diagnostics,
            runtime=self._runtime,
            ops=ops,
            phase_overrides=phase_overrides,
        )
        return self._run_locked(context, mode=mode, cache=cache, ops=ops)

    def _run_locked(
        self,
        context: StageContext,
        *,
        mode: EnhancementMode,
        cache: SharedStageCache | None,
        ops,
    ) -> PipelineResult:
        watches = Stopwatch()
        reports: list[StageReport] = []

        failed: str | None = None

        for stage in self.stages_for(mode):
            if stage.status is not StageStatus.IMPLEMENTED:
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        detail=self._pending_reason(stage),
                    )
                )
                continue

            cached = self._cached(cache, stage)
            if cached is not None:
                if cached.error is not None:
                    if isinstance(cached.error, InferenceError):
                        raise cached.error
                    failed = f"The '{stage.name.value}' stage failed: {cached.error}"
                    watches.add(stage.name.value, cached.duration_ms)
                    reports.append(
                        StageReport(
                            name=stage.name,
                            status=stage.status,
                            duration_ms=cached.duration_ms,
                            detail=failed,
                        )
                    )
                    break
                cache.restore(stage.name, context)  # type: ignore[union-attr]
                watches.add(stage.name.value, cached.duration_ms)
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        duration_ms=cached.duration_ms,
                    )
                )
                continue

            try:
                with watches.measure(stage.name.value):
                    stage.run(context)
                    if ops is not None:
                        context.runtime.synchronize()
            except StageNotImplemented as exc:
                reports.append(
                    StageReport(name=stage.name, status=stage.status, detail=str(exc))
                )
            except InferenceError as exc:
                self._record_error(cache, stage, watches, exc)
                raise
            except Exception as exc:
                logger.exception("Pipeline stage %s failed", stage.name.value)
                failed = f"The '{stage.name.value}' stage failed: {exc}"
                self._record_error(cache, stage, watches, exc)
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        duration_ms=watches.durations_ms.get(stage.name.value),
                        detail=failed,
                    )
                )
                break
            else:
                self._record_success(cache, stage, watches, context)
                reports.append(
                    StageReport(
                        name=stage.name,
                        status=stage.status,
                        duration_ms=watches.durations_ms.get(stage.name.value),
                    )
                )

        artifacts = self._artifacts(context.outputs)
        final_image = context.outputs.get(ArtifactKind.FINAL_OUTPUT)

        status, message = self._outcome(reports, failed, mode)
        if status is ProcessingStatus.COMPLETED and final_image is None:
            status = ProcessingStatus.FAILED
            message = "The pipeline completed without producing a final image."

        return PipelineResult(
            status=status,
            mode=mode,
            stages=tuple(reports),
            artifacts=artifacts,
            timings_ms=dict(watches.durations_ms),
            total_ms=watches.total_ms,
            final_image=final_image,
            message=message,
            diagnostics=dict(context.diagnostics)
            if context.capture_diagnostics
            else {},
            diagnostic_images=dict(context.diagnostic_images)
            if context.capture_diagnostics
            else {},
        )

    # -- cache helpers -----------------------------------------------------

    @staticmethod
    def _cached(
        cache: SharedStageCache | None, stage: PipelineStage
    ) -> CachedStage | None:
        if cache is None or stage.name not in SHARED_STAGES:
            return None
        return cache.get(stage.name)

    @staticmethod
    def _record_success(
        cache: SharedStageCache | None,
        stage: PipelineStage,
        watches: Stopwatch,
        context: StageContext,
    ) -> None:
        if cache is None or stage.name not in SHARED_STAGES:
            return
        cache.record(
            stage.name,
            duration_ms=watches.durations_ms.get(stage.name.value, 0.0),
            context=context,
            produced=stage.produces,
        )

    @staticmethod
    def _record_error(
        cache: SharedStageCache | None,
        stage: PipelineStage,
        watches: Stopwatch,
        error: BaseException,
    ) -> None:
        if cache is None or stage.name not in SHARED_STAGES:
            return
        cache.record_error(
            stage.name,
            duration_ms=watches.durations_ms.get(stage.name.value, 0.0),
            error=error,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _pending_reason(stage: PipelineStage) -> str:
        return getattr(
            stage,
            "pending_reason",
            f"The '{stage.name.value}' stage is not implemented yet.",
        )

    def artifact_capabilities(self) -> tuple[Artifact, ...]:
        """Describe every planned artifact before a run (nothing produced)."""
        return self._artifacts({})

    def _artifacts(
        self, outputs: dict[ArtifactKind, Image.Image]
    ) -> tuple[Artifact, ...]:
        artifacts: list[Artifact] = []
        for descriptor in PLANNED_ARTIFACTS:
            image = outputs.get(descriptor.kind)
            if image is not None:
                artifacts.append(
                    Artifact(descriptor=descriptor, available=True, image=image)
                )
                continue

            producer = self._by_name.get(descriptor.produced_by)
            if producer is None:
                reason = "No stage produces this output yet."
            elif producer.status is StageStatus.IMPLEMENTED:
                reason = "This output was not produced for this image."
            else:
                reason = self._pending_reason(producer)
            artifacts.append(
                Artifact(descriptor=descriptor, available=False, reason=reason)
            )

        return tuple(artifacts)

    @staticmethod
    def _outcome(
        reports: list[StageReport], failed: str | None, mode: EnhancementMode
    ) -> tuple[ProcessingStatus, str]:
        if failed is not None:
            return ProcessingStatus.FAILED, failed

        pending = [
            report.name.value
            for report in reports
            if report.status is not StageStatus.IMPLEMENTED
        ]
        if pending:
            message = (
                "Enhancement is not available yet. Pending stages for "
                f"'{MODE_LABELS[mode]}': {', '.join(pending)}. "
                "No output was generated."
            )
            return ProcessingStatus.UNAVAILABLE, message

        return ProcessingStatus.COMPLETED, "Enhancement completed."

