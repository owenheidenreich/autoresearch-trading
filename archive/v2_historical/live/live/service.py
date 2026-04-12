"""Main service loop: market data -> features -> model -> TradeIntent -> execute.

Orchestrates the bar-by-bar trading loop during RTH (9:30-16:00 ET).
Connects market data stream, decision engine, and execution engine.

v1 origin: training/live/service.py (PaperTradingService, PaperLiveConfig)
"""
# TODO: TradingService class
# TODO: ServiceConfig dataclass (host, port, model_path, etc.)
# TODO: Main loop: poll bar -> compute features -> run model -> execute
# TODO: Graceful shutdown, EOD flatten, kill switch
# TODO: Daily summary (session stats, trade log, P&L)
