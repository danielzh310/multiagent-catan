from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from core.constants import PlayerId


class TurnPhase(IntEnum):
    SETUP = 0
    ROLL = 1
    MAIN_ACTION = 2
    TRADE_PROPOSE = 3
    TRADE_RESPOND = 4
    END_TURN = 5
    GAME_OVER = 6


class ControllerType(IntEnum):
    GAMEPLAY = 0
    TRADE = 1
    ENVIRONMENT = 2
    NONE = 3


@dataclass
class PhaseDecision:
    phase: TurnPhase
    controller: ControllerType
    acting_player: Optional[PlayerId]
    target_player: Optional[PlayerId] = None


class PhaseRouter:
    """
    Decides which policy acts at each point in the turn.

    Gameplay model handles:
    - setup placements
    - board progression actions
    - end-turn choices

    Trade model handles:
    - trade proposals
    - trade responses / counteroffers
    """

    def __init__(self):
        self.current_phase = TurnPhase.SETUP
        self.phase_index = 0

    def reset(self) -> None:
        self.current_phase = TurnPhase.SETUP
        self.phase_index = 0

    def get_phase(self) -> TurnPhase:
        return self.current_phase

    def set_phase(self, phase: TurnPhase) -> None:
        self.current_phase = phase

    def next_phase(self, engine) -> TurnPhase:
        """
        Advance phase based on current engine state.
        """
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return self.current_phase

        if engine.initial_placement_phase:
            self.current_phase = TurnPhase.SETUP
            return self.current_phase

        if engine.robber_pending:
            self.current_phase = TurnPhase.MAIN_ACTION
            return self.current_phase

        if engine.trade_manager.has_pending_trade():
            self.current_phase = TurnPhase.TRADE_RESPOND
            return self.current_phase

        if self.current_phase == TurnPhase.ROLL:
            self.current_phase = TurnPhase.MAIN_ACTION
        elif self.current_phase == TurnPhase.MAIN_ACTION:
            self.current_phase = TurnPhase.TRADE_PROPOSE
        elif self.current_phase == TurnPhase.TRADE_PROPOSE:
            self.current_phase = TurnPhase.END_TURN
        elif self.current_phase == TurnPhase.END_TURN:
            self.current_phase = TurnPhase.ROLL
        else:
            self.current_phase = TurnPhase.ROLL

        self.phase_index += 1
        return self.current_phase

    def get_controller(self, engine) -> PhaseDecision:
        """
        Return which model should act right now.
        """
        if engine.winner is not None:
            return PhaseDecision(
                phase=TurnPhase.GAME_OVER,
                controller=ControllerType.NONE,
                acting_player=None,
            )

        if engine.initial_placement_phase:
            return PhaseDecision(
                phase=TurnPhase.SETUP,
                controller=ControllerType.GAMEPLAY,
                acting_player=engine.get_current_player_id(),
            )

        if engine.trade_manager.has_pending_trade():
            pending = engine.trade_manager.get_pending_trade()
            return PhaseDecision(
                phase=TurnPhase.TRADE_RESPOND,
                controller=ControllerType.TRADE,
                acting_player=pending.target,
                target_player=pending.proposer,
            )

        if self.current_phase == TurnPhase.ROLL:
            return PhaseDecision(
                phase=TurnPhase.ROLL,
                controller=ControllerType.ENVIRONMENT,
                acting_player=engine.get_current_player_id(),
            )

        if self.current_phase == TurnPhase.MAIN_ACTION:
            return PhaseDecision(
                phase=TurnPhase.MAIN_ACTION,
                controller=ControllerType.GAMEPLAY,
                acting_player=engine.get_current_player_id(),
            )

        if self.current_phase == TurnPhase.TRADE_PROPOSE:
            return PhaseDecision(
                phase=TurnPhase.TRADE_PROPOSE,
                controller=ControllerType.TRADE,
                acting_player=engine.get_current_player_id(),
            )

        if self.current_phase == TurnPhase.END_TURN:
            return PhaseDecision(
                phase=TurnPhase.END_TURN,
                controller=ControllerType.GAMEPLAY,
                acting_player=engine.get_current_player_id(),
            )

        return PhaseDecision(
            phase=self.current_phase,
            controller=ControllerType.NONE,
            acting_player=None,
        )

    def begin_turn(self, engine) -> None:
        """
        Called once at the start of a non-setup turn.
        """
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.initial_placement_phase:
            self.current_phase = TurnPhase.SETUP
            return

        self.current_phase = TurnPhase.ROLL

    def complete_roll_phase(self, engine) -> None:
        """
        Called after the environment resolves dice / robber trigger.
        """
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.trade_manager.has_pending_trade():
            self.current_phase = TurnPhase.TRADE_RESPOND
            return

        self.current_phase = TurnPhase.MAIN_ACTION

    def complete_main_action_phase(self, engine) -> None:
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.trade_manager.has_pending_trade():
            self.current_phase = TurnPhase.TRADE_RESPOND
            return

        self.current_phase = TurnPhase.TRADE_PROPOSE

    def complete_trade_propose_phase(self, engine) -> None:
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.trade_manager.has_pending_trade():
            self.current_phase = TurnPhase.TRADE_RESPOND
            return

        self.current_phase = TurnPhase.END_TURN

    def complete_trade_respond_phase(self, engine) -> None:
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.trade_manager.has_pending_trade():
            self.current_phase = TurnPhase.TRADE_RESPOND
            return

        self.current_phase = TurnPhase.END_TURN

    def complete_end_turn_phase(self, engine) -> None:
        if engine.winner is not None:
            self.current_phase = TurnPhase.GAME_OVER
            return

        if engine.initial_placement_phase:
            self.current_phase = TurnPhase.SETUP
            return

        self.current_phase = TurnPhase.ROLL