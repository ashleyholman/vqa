import React, { useState, useEffect } from 'react';
import { Link, useLocation, useParams, useRoutes } from 'react-router-dom';
import Config from './Config';
import { ErrorAnalysisContext } from '../contexts/ErrorAnalysisContext.js';

import AnswerClassSampleViewer from './AnswerClassSampleViewer.js';
import ErrorAnalysisTable from './ErrorAnalysisTable.js';
import MetricsChartGrid from './MetricsChartGrid.js';
import './RunDetails.css';

function RunDetails() {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);
  const { setIsErrorAnalysisSummaryDataLoaded, setErrorAnalysisSummaryData } = React.useContext(ErrorAnalysisContext);
  const [tab, setTab] = useState('charts');
  const location = useLocation();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/data/run/${runId}/main.json`);
        const data = await response.json();
        setRunData(data);

        if (data.has_error_analysis) {
          console.log("Fetching error analysis data")
          const responseErrorAnalysis = await fetch(`/data/run/${runId}/error_analysis/summary.json`);
          const summaryData = await responseErrorAnalysis.json();
          setErrorAnalysisSummaryData(summaryData);
          setIsErrorAnalysisSummaryDataLoaded(true);
        } else {
          console.log("NOT fetching error analysis data for this run")
        }
      } catch (error) {
        console.error('Error:', error);
      }
    };

    fetchData();
  }, [runId]);

  useEffect(() => {
    const pathParts = location.pathname.split('/').filter(Boolean);

    if (pathParts.length > 2) {
      const thirdPart = pathParts[2];

      switch(thirdPart) {
        case 'config':
          setTab('config');
          break;
        case 'charts':
          setTab('charts');
          break;
        case 'error_analysis':
          setTab('error_analysis');
          break;
        default:
          setTab('charts');
      }
    } else {
      setTab('charts'); // Default to 'charts' for /run/:runId/
    }
  }, [location]);

  let config = null;
  let isMiniRun = null;
  let hasErrorAnalysis = null;
  let metrics = null;

  if (runData) {
    config = runData['config'];
    isMiniRun = runData['validation_dataset_type'] === 'mini';
    hasErrorAnalysis = runData['has_error_analysis'] === true;
    metrics = runData.metrics;
  }

  let element = useRoutes([
    { path: '', element: <MetricsChartGrid metrics={metrics} isMiniRun={isMiniRun} /> },
    { path: 'config', element: <Config configData={config} /> },
    { path: 'charts', element: <MetricsChartGrid metrics={metrics} isMiniRun={isMiniRun} /> },
    { path: 'error_analysis', element: <ErrorAnalysisTable runId={runId} /> },
    { path: 'error_analysis/:classId/*', element: <AnswerClassSampleViewer runId={runId} /> }
  ]);

  if (!runData) {
    return <div>Loading...</div>;
  }

  const isErrorAnalysisRoute = location.pathname.includes('/error_analysis/');

  return (
    <div className="run-details">
      {!isErrorAnalysisRoute && (
        <div className="tab-bar">{'[ '}
          <Link to={`/run/${runId}/config`} className={tab === 'config' ? 'active-tab' : ''} onClick={(e) => {setTab('config')}}>Config</Link>
          {' | '}
          <Link to={`/run/${runId}/charts`} className={tab === 'charts' ? 'active-tab' : ''} onClick={(e) => {setTab('charts')}}>Charts</Link>
          {hasErrorAnalysis && (
            <>
              {' | '}
              <Link to={`/run/${runId}/error_analysis`} className={tab === 'error_analysis' ? 'active-tab' : ''} onClick={(e) => {setTab('error_analysis')}}>
                Error Analysis
              </Link>
            </>
          )}
          {' ] '}
        </div>
      )}
      {!isErrorAnalysisRoute && (
        <h2 className="run-details-title">{config['model_name']}</h2>
      )}
      {element}
    </div>
  );
}

export default RunDetails;