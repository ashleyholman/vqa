import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { ErrorAnalysisContext } from '../contexts/ErrorAnalysisContext.js';
import { AiOutlineArrowUp, AiOutlineArrowDown } from 'react-icons/ai'; // import icons

function ErrorAnalysisTable({ runId }) {
  const { errorAnalysisSummaryData, isErrorAnalysisSummaryDataLoaded } = React.useContext(ErrorAnalysisContext);
  const [sortField, setSortField] = useState('class_id');
  const [sortOrder, setSortOrder] = useState('asc');
  const [sortedKeys, setSortedKeys] = useState([]);

  useEffect(() => {
    if (isErrorAnalysisSummaryDataLoaded) {
      let keys = Object.keys(errorAnalysisSummaryData);
      keys.sort((a, b) => {
        if (sortField === 'class_id') {
          return sortOrder === 'asc' ? Number(a) - Number(b) : Number(b) - Number(a);
        } else if (sortField === 'label') {
          const labelA = errorAnalysisSummaryData[a].class_label;
          const labelB = errorAnalysisSummaryData[b].class_label;
          return sortOrder === 'asc' ? labelA.localeCompare(labelB) : labelB.localeCompare(labelA);
        } else {
          // adjust this section based on the possible sortField value
          const valA = errorAnalysisSummaryData[a].statistics[sortField];
          const valB = errorAnalysisSummaryData[b].statistics[sortField];
          return sortOrder === 'asc' ? valA - valB : valB - valA;
        }
      });
      setSortedKeys(keys);
    }
  }, [isErrorAnalysisSummaryDataLoaded, errorAnalysisSummaryData, sortField, sortOrder]);

  const handleSort = (field) => {
    if (field === sortField) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  const renderSortArrow = (field) => {
    if (sortField === field) {
      return sortOrder === 'asc' ? <AiOutlineArrowUp /> : <AiOutlineArrowDown />;
    }
    return <AiOutlineArrowDown className="hidden-icon" />;
  };

  if (!isErrorAnalysisSummaryDataLoaded) {
    return <div>Loading error analysis data...</div>;
  }

  // Calculate total data size
  const totalDataSize = sortedKeys.reduce((total, key) => {
    return total + errorAnalysisSummaryData[key].statistics.TP + errorAnalysisSummaryData[key].statistics.FN;
  }, 0);

  const headings = {
    'class_id': 'Class ID',
    'label': 'Label',
    'actualPositives': 'Actual Positives (%)',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1Score': 'F1 Score',
    'TP': 'True Positives',
    'FP': 'False Positives',
    'FN': 'False Negatives',
    'sampleQuestions': 'Sample Questions',
  };
  
  return (
    <div className="table-responsive">
      <table className="table table-dark">
        <thead>
          <tr>
            {Object.entries(headings).map(([key, label]) => (
              <th key={key} onClick={key !== 'sampleQuestions' ? () => handleSort(key) : undefined}>
                <div className="header-div">
                  {label} {key !== 'sampleQuestions' ? renderSortArrow(key) : null}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sortedKeys.map(key => {
            const TP = errorAnalysisSummaryData[key].statistics.TP;
            const FP = errorAnalysisSummaryData[key].statistics.FP;
            const FN = errorAnalysisSummaryData[key].statistics.FN;
            const actualPositives = errorAnalysisSummaryData[key].statistics.actualPositives;
            const precision = errorAnalysisSummaryData[key].statistics.precision.toFixed(2);
            const recall = errorAnalysisSummaryData[key].statistics.recall.toFixed(2);
            const f1Score = errorAnalysisSummaryData[key].statistics.f1Score.toFixed(2);

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