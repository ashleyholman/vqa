import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Config from './Config';
import MetricsChartGrid from './MetricsChartGrid.js';
import './RunDetails.css';

function RunDetails() {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/data/run/${runId}/main.json`);
        const data = await response.json();
        setRunData(data);
      } catch (error) {
        console.error('Error:', error);
      }
    };

    fetchData();
  }, [runId]);

  if (!runData) {
    return <div>Loading...</div>;
  }

  const config = runData['config'];
  const isMiniRun = runData['validation_dataset_type'] === 'mini';

  return (
    <div className="run-details">
      <h2 className="run-details-title">{config['model_name']}</h2>
      <Config configData={config} />
      <MetricsChartGrid metrics={runData.metrics} isMiniRun={isMiniRun} />
    </div>
  );
}

export default RunDetails;