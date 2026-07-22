from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    """Policy controlling the provider-tool loop for one agent."""

    max_steps: int = 20
    max_no_tool_responses: int = 3
    max_repeated_tool_calls: int = 3
    max_run_tokens: int | None = None
    require_explicit_finish: bool = False
    parallel_tool_calls: bool = False
    max_parallel_tool_calls: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.parallel_tool_calls, bool):
            raise TypeError("parallel_tool_calls must be a boolean")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be greater than zero")
        if self.max_no_tool_responses <= 0:
            raise ValueError("max_no_tool_responses must be greater than zero")
        if self.max_repeated_tool_calls <= 0:
            raise ValueError("max_repeated_tool_calls must be greater than zero")
        if self.max_run_tokens is not None and self.max_run_tokens <= 0:
            raise ValueError("max_run_tokens must be greater than zero")
        if self.max_parallel_tool_calls is not None:
            if isinstance(self.max_parallel_tool_calls, bool) or not isinstance(
                self.max_parallel_tool_calls, int
            ):
                raise TypeError("max_parallel_tool_calls must be an integer")
            if self.max_parallel_tool_calls <= 0:
                raise ValueError("max_parallel_tool_calls must be greater than zero")
