import React from 'react';
import { useLocation } from 'react-router-dom';
import { Link } from 'react-router-dom';
import { ErrorAnalysisContext } from '../contexts/ErrorAnalysisContext.js';

function Breadcrumbs() {
  const location = useLocation();
  const pathParts = location.pathname.split('/').filter(Boolean);
  const { errorAnalysisSummaryData, isErrorAnalysisSummaryDataLoaded } = React.useContext(ErrorAnalysisContext);

  let breadcrumbs = null;

  if (pathParts.length === 0) {
    breadcrumbs = <Link to="/" style={{ color: '#00A878', fontWeight: 'bold' }}>Runs</Link>;
  } else if (pathParts[0] === 'run' && pathParts.length >= 2) {
    const runId = pathParts[1];
    breadcrumbs = (
      <>
        <Link to="/" style={{ color: '#00A878', fontWeight: 'bold' }}>Runs</Link>{' -> '}
        <Link to={`/run/${runId}`} style={{ color: '#00A878', fontWeight: 'bold' }}>{runId}</Link>
      </>
    );

    if (pathParts[2] === 'config') {
      breadcrumbs = (
        <>
          {breadcrumbs}{' -> '} Config
        </>
      );
    } else if (pathParts[2] === 'charts') {
      breadcrumbs = (
        <>
          {breadcrumbs}{' -> '} Charts
        </>
      );
    } else if (pathParts[2] === 'error_analysis') {
      breadcrumbs = (
        <>
          {breadcrumbs}{' -> '}
          <Link to={`/run/${runId}/error_analysis`} style={{ color: '#00A878', fontWeight: 'bold' }}>Error Analysis</Link>
        </>
      );
      if (pathParts.length >= 4) {
        const classId = pathParts[3];
        let className = `Class ${classId}`
        if (isErrorAnalysisSummaryDataLoaded) {
          className = errorAnalysisSummaryData[classId]?.class_label ? `Class "${errorAnalysisSummaryData[classId].class_label}"` : `Class ${classId}`;
        }
        breadcrumbs = (
          <>
            {breadcrumbs}{' -> '}
            <Link to={`/run/${runId}/error_analysis/${classId}`} style={{ color: '#00A878', fontWeight: 'bold' }}>{className}</Link>
          </>
        );
        if (pathParts.length >= 5) {
          const categoryMap = {
            'tp': 'True Positives',
            'fp': 'False Positives',
            'fn': 'False Negatives',
            'tn': 'True Negatives',
          };
          const category = categoryMap[pathParts[4]];
          breadcrumbs = (
            <>
              {breadcrumbs}{' -> '} {category}
            </>
          );
        }
      }
    }
  }
  
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'flex-start',
      alignItems: 'center',
      padding: '5px 10px',
      backgroundColor: '#333'
    }}>
      <div style={{ color: 'white', lineHeight: '1.2' }}>{breadcrumbs}</div>
    </div>
  );
}

export default Breadcrumbs;