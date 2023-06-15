import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Config from './Config';
import MetricsChartGrid from './MetricsChartGrid.js';

function RunDetails() {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(`/data/run_${runId}.json`);
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

  return (
    <div style={{ margin: '0 auto', padding: '0 20px', maxWidth: '95%' }}>
      <h2 style={{ overflowWrap: 'anywhere' }}>{config['model_name']}</h2>
      <Config configData={config} />
      <MetricsChartGrid metrics={runData.metrics} />
    </div>
  );
}

export default RunDetails;