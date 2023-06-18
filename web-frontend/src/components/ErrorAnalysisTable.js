import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import './ErrorAnalysisTable.css';

function ErrorAnalysisTable({ runId, errorAnalysisSummaryData }) {
  if (!errorAnalysisSummaryData) {
    return <div>Loading error analysis data...</div>;
  }

  const sortedKeys = Object.keys(errorAnalysisSummaryData).sort((a, b) => Number(a) - Number(b));

  return (
    <div className="table-responsive">
      <table className="table table-dark">
        <thead>
          <tr>
            <th>Class ID</th>
            <th>Label</th>
            <th>True Positives</th>
            <th>False Positives</th>
            <th>False Negatives</th>
            <th>Sample Questions</th>
          </tr>
        </thead>
        <tbody>
          {sortedKeys.map(key => (
            <tr key={key}>
              <td>{key}</td>
              <td>{errorAnalysisSummaryData[key].class_label}</td>
              <td>{errorAnalysisSummaryData[key].statistics.TP}</td>
              <td>{errorAnalysisSummaryData[key].statistics.FP}</td>
              <td>{errorAnalysisSummaryData[key].statistics.FN}</td>
              <td>
                  <Link to={`/run/${runId}/error_analysis/${key}`}>
                    View
                  </Link>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ErrorAnalysisTable;