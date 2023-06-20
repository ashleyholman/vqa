import React, { useState } from 'react';
export const ErrorAnalysisContext = React.createContext();

export const ErrorAnalysisProvider = ({ children }) => {
  const [errorAnalysisSummaryData, setErrorAnalysisSummaryData] = useState(null);
  const [isErrorAnalysisSummaryDataLoaded, setIsErrorAnalysisSummaryDataLoaded] = useState(false);

  const value = {
    errorAnalysisSummaryData,
    setErrorAnalysisSummaryData: (data) => {
      // Augment the data with some additional derived statistics
      Object.entries(data).forEach(([key, item]) => {
        const TP = item.statistics.TP;
        const FP = item.statistics.FP;
        const FN = item.statistics.FN;

        const actualPositives = TP + FN;
        const precision = TP + FP === 0 ? 0 : TP / (TP + FP);
        const recall = TP + FN === 0 ? 0 : TP / (TP + FN);
        const f1Score = (precision + recall) === 0 ? 0 : 2 * ((precision * recall) / (precision + recall));

        data[key].statistics = { ...item.statistics, actualPositives, precision, recall, f1Score };
      });

      setErrorAnalysisSummaryData(data);
    },
    isErrorAnalysisSummaryDataLoaded,
    setIsErrorAnalysisSummaryDataLoaded,
  };

  return (
    <ErrorAnalysisContext.Provider value={value}>
      {children}
    </ErrorAnalysisContext.Provider>
  );
};
