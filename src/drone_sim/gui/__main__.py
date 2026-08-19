# src/drone_sim/gui/__main__.py
import os
os.environ["QT_API"] = "pyside6"  # MUST precede all matplotlib/PySide6 imports

import logging
import sys

from PySide6.QtWidgets import QApplication

from drone_sim.gui.main_window import MainWindow


def _configure_logging() -> None:
   """Wire up a console logger so prediction-layer activity (LSTM/BoF) is visible.

   Defaults to WARNING globally and INFO for ``drone_sim.prediction`` so the
   BoF adapter and provider chatter shows up while solver/inner-loop noise
   stays quiet. Override with ``DRONE_SIM_LOG_LEVEL`` (root) or
   ``BOF_LOG_LEVEL`` (prediction-only) — values are standard names like
   DEBUG, INFO, WARNING.
   """
   root_level = os.environ.get("DRONE_SIM_LOG_LEVEL", "INFO").upper()
   pred_level = os.environ.get("BOF_LOG_LEVEL", "INFO").upper()
   logging.basicConfig(
      level=getattr(logging, root_level, logging.WARNING),
      format="%(asctime)s %(name)s %(levelname)s %(message)s",
      datefmt="%H:%M:%S",
   )
   logging.getLogger("drone_sim.prediction").setLevel(
      getattr(logging, pred_level, logging.INFO)
   )


def main() -> None:
   _configure_logging()
   # Use QApplication.instance() guard — never assume no QApplication exists
   # (locked decision: some IDEs already own the event loop)
   app = QApplication.instance() or QApplication(sys.argv)
   window = MainWindow()
   window.show()
   sys.exit(app.exec())


if __name__ == "__main__":
   main()
