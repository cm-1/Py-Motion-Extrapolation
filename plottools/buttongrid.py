from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

@dataclass
class ButtonGridConfig:
    """Configuration for ButtonGrid appearance and layout."""
    cell_width: float = 0.08      # Width of each button
    cell_height: float = 0.04     # Height of each button
    button_on_color: str = 'lightgreen'
    button_off_color: str = 'lightcoral'
    header_color: str = 'white'   # Color for header buttons
    # For now, we'll use margins as a way to put the grid in either the
    # top left, bottom centre, etc. That is, if one of the margins for a
    # direction is not None, we'll use it to set the table that distance away
    # from the respective edge. And if both are None, we'll centre it.
    margin_bottom: float = 0.01
    margin_top: float = None
    margin_left: float = None
    margin_right: float = None

class ButtonGrid:
    def __init__(
        self,
        fig: plt.Figure,
        column_labels: List[str],
        row_labels: Optional[List[str]] = None,
        config: Optional[ButtonGridConfig] = None
    ):
        """
        Initialize a ButtonGrid instance.
        
        Args:
            fig: matplotlib figure to attach the buttons to
            column_labels: list of labels for columns
            row_labels: optional list of labels for rows. If None, creates a single row
            config: ButtonGridConfig instance for customization
        """
        self.fig = fig
        self.column_labels = column_labels
        self.row_labels = row_labels if row_labels is not None else []
        self.config = config or ButtonGridConfig()

        total_w = self.config.cell_width * len(self.column_labels)
        total_h = self.config.cell_height * max(len(self.row_labels), 1)

        self.left_x = ButtonGrid._dist_from_edge(
            self.config.margin_left, self.config.margin_right, total_w
        )
        self.top_y = 1.0 - ButtonGrid._dist_from_edge(
            self.config.margin_top, self.config.margin_bottom, total_h
        )

        # Initialize buttons list and states array
        self.buttons = []
        rows = len(self.row_labels) if self.row_labels else 1
        self.button_states = np.full((rows, len(column_labels)), False)
        
        # Callback function
        self._toggle_callback = None

        self._create_grid()
    
    def _dist_from_edge(near_margin: float, far_margin: float, span: float):
        dist = near_margin # Just return near_margin if it is not None.
        if dist is None:
            if far_margin is None:
                dist = 0.5 - (span / 2.0) # Centre the object.
            else:
                dist = 1.0 - (far_margin + span) # 1.0 - (dist from far edge).
        return dist
    
    def _get_button_axes(self, row_ind: int, col_ind: int):
        """Calculate the position for each button."""
        return plt.axes((
            self.left_x + self.config.cell_width * col_ind,
            self.top_y - self.config.cell_height * row_ind,
            self.config.cell_width,
            self.config.cell_height
        ))

    def on_button_clicked(self, row: int, col: int, button: Button):
        """Internal button click handler."""
        
        row_is_inner = (row > 0) or (not self.row_labels) 
        if row_is_inner and col > 0:            
            # Regular button click; need to ignore the header col/row, as they will 
            # not appear in the state itself.
            # TODO: Add option on whether or not to create buttons for the headers 
            # or to just create text instead!
            state_row, state_col = row - 1, col - 1
            if not self.row_labels:
                state_row = 0

            new_state = not self.button_states[state_row, state_col]
            self.button_states[state_row, state_col] = new_state
            
            # Update button appearance
            button.color = (
                self.config.button_off_color, self.config.button_on_color
            )[new_state]
            
            # Call user callback if defined
            if self._toggle_callback:
                self._toggle_callback(state_row, state_col, new_state)
        else:
            # TODO: Handle clicks to header buttons
            ...

    def _create_grid(self):
        """Create and set up the button grid."""
        cols = len(self.column_labels) + 1
        rows = 1 if (not self.row_labels) else len(self.row_labels) + 1
            
        for r_ind in range(rows):
            button_row = []
            for c_ind in range(cols):
                btn_ax = self._get_button_axes(r_ind, c_ind)
                row_is_inner = (r_ind > 0) or (not self.row_labels)

                # Determine button text and color
                text = ""
                btn_color = self.config.header_color

                if row_is_inner and c_ind > 0:
                    btn_color = self.config.button_off_color

                if c_ind > 0 and r_ind == 0:
                    text = self.column_labels[c_ind - 1]
                elif r_ind > 0 and c_ind == 0:
                    text = self.row_labels[r_ind - 1]                    
                
                button = Button(btn_ax, text, color=btn_color)
                button.on_clicked(
                    lambda event, row=r_ind, col=c_ind, btn=button: 
                    self.on_button_clicked(row, col, btn)
                )
                button_row.append(button)
            self.buttons.append(button_row)
    
    def set_toggle_callback(self, callback: Callable[[int, int, bool], None]):
        """
        Set callback for regular button toggles.
        
        Args:
            callback: function taking (row, col, new_state) as arguments
        """
        self._toggle_callback = callback
    
    def get_active_indices(self) -> List[tuple]:
        """Return list of (row, col) tuples for active buttons."""
        return list(map(tuple, np.argwhere(self.button_states)))