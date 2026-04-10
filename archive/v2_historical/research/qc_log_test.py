# Log test - which logging method shows output?
from AlgorithmImports import *


class LogTest(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2025, 12, 15)
        self.SetEndDate(2025, 12, 16)
        self.SetCash(100000)
        self.spx = self.AddIndex("SPX", Resolution.Minute)
        self.done = False
        self.Log("LOG from Initialize")
        self.Debug("DEBUG from Initialize")

    def OnData(self, data):
        if self.done:
            return
        self.done = True
        self.Log("LOG from OnData")
        self.Debug("DEBUG from OnData")
        self.Error("ERROR from OnData")

    def OnEndOfAlgorithm(self):
        self.Log("LOG from OnEndOfAlgorithm")
        self.Debug("DEBUG from OnEndOfAlgorithm")
