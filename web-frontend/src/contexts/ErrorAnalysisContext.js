import React, { useState } from 'react';

export const ErrorAnalysisContext = React.createContext();

export const ErrorAnalysisProvider = ({ children }) => {
  const [errorAnalysisSummaryData, setErrorAnalysisSummaryData] = useState(null);
  const [isErrorAnalysisSummaryDataLoaded, setIsErrorAnalysisSummaryDataLoaded] = useState(false);

  const value = {
    errorAnalysisSummaryData,
    setErrorAnalysisSummaryData,
    isErrorAnalysisSummaryDataLoaded,
    setIsErrorAnalysisSummaryDataLoaded,
  };

  return (
    <ErrorAnalysisContext.Provider value={value}>
      {children}
    </ErrorAnalysisContext.Provider>
  );
};