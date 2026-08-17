from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table
from rich.text import Text


class TerminalDashboard:
    def __init__(self, robot: str, policy: str) -> None:
        self._robot = robot
        self._policy = policy
        self._console = Console()
        self._live: Optional[Live] = None

    def start(self) -> None:
        if not self._console.is_terminal:
            return
        self._live = Live(
            Text("Starting controller dashboard..."),
            console=self._console,
            auto_refresh=False,
            screen=True,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start(refresh=True)

    def update(
        self,
        state: str,
        phase: str,
        progress: float,
        command_source: str,
        command: tuple,
        sensor_health: str,
        last_key: str,
        policy_enabled: bool,
    ) -> None:
        if self._live is None:
            return

        progress_bar = Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=32),
            TaskProgressColumn(),
            expand=True,
        )
        progress_bar.add_task(phase, total=100, completed=round(progress * 100))

        state_color = {
            "passive": "yellow",
            "get_up": "cyan",
            "locomotion": "green",
            "get_down": "magenta",
        }.get(state, "white")
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold", width=13)
        table.add_column()
        table.add_column(style="bold", width=13)
        table.add_column()
        table.add_row(
            "State",
            Text(state.upper(), style=state_color),
            "Policy",
            f"{self._policy} ({'enabled' if policy_enabled else 'standby'})",
        )
        table.add_row(
            "Command",
            f"{command_source}: vx={command[0]:+0.2f}, vy={command[1]:+0.2f}, yaw={command[2]:+0.2f}",
            "Last key",
            last_key,
        )
        table.add_row("Sensors", sensor_health, "Robot", self._robot)
        help_text = Text("0 get up  •  1 locomotion  •  9 get down  •  P passive  •  W/S/A/D/Q/E command  •  Space clear  •  N cmd_vel")
        renderable = Panel(
            Group(progress_bar, table, help_text),
            title="[bold]RL Policy Runtime[/bold]",
            border_style=state_color,
            box=box.ROUNDED,
        )
        self._live.update(renderable, refresh=True)

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
