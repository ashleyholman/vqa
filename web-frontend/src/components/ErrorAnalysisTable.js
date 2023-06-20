import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ErrorAnalysisContext } from '../contexts/ErrorAnalysisContext.js';

function ErrorAnalysisTable({ runId }) {
  const { errorAnalysisSummaryData, isErrorAnalysisSummaryDataLoaded } = React.useContext(ErrorAnalysisContext);

  if (!isErrorAnalysisSummaryDataLoaded) {
    return <div>Loading error analysis data...</div>;
  }

  const sortedKeys = Object.keys(errorAnalysisSummaryData).sort((a, b) => Number(a) - Number(b));

  // Calculate total data size
  const totalDataSize = sortedKeys.reduce((total, key) => {
    return total + errorAnalysisSummaryData[key].statistics.TP + errorAnalysisSummaryData[key].statistics.FN;
  }, 0);

  return (
    <div className="table-responsive">
      <table className="table table-dark">
        <thead>
          <tr>
            <th>Class ID</th>
            <th>Label</th>
            <th>Actual Positives (%)</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 Score</th>
            <th>True Positives</th>
            <th>False Positives</th>
            <th>False Negatives</th>
            <th>Sample Questions</th>
          </tr>
        </thead>
        <tbody>
          {sortedKeys.map(key => {
            const TP = errorAnalysisSummaryData[key].statistics.TP;
            const FP = errorAnalysisSummaryData[key].statistics.FP;
            const FN = errorAnalysisSummaryData[key].statistics.FN;

            const actualPositives = TP + FN;
            const precision = TP + FP === 0 ? 0 : (TP / (TP + FP)).toFixed(2);
            const recall = TP + FN === 0 ? 0 : (TP / (TP + FN)).toFixed(2);
            const f1Score = (precision + recall) === 0 ? 0 : (2 * ((precision * recall) / (precision + recall))).toFixed(2);

            return (
              <tr key={key}>
                <td>{key}</td>
                <td>{errorAnalysisSummaryData[key].class_label}</td>
                <td>{`${actualPositives} (${((actualPositives / totalDataSize) * 100).toFixed(2)}%)`}</td>
                <td>{precision}</td>
                <td>{recall}</td>
                <td>{f1Score}</td>
                <td>{TP}</td>
                <td>{FP}</td>
                <td>{FN}</td>
                <td>
                  <Link to={`/run/${runId}/error_analysis/${key}`}>
                    View
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default ErrorAnalysisTable;