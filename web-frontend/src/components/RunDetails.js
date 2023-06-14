import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Config from './Config';
import MetricChart from './MetricChart';

function RunDetails() {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);

  const metricsList = ['accuracy', 'top_5_accuracy', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'f1_score_macro', 'f1_score_micro'];

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

  const metrics = runData.metrics;
  const chartData = Object.keys(metrics).map((key) => ({
    epoch: metrics[key].epoch,
    ...metrics[key] // Spread remaining properties
  }));

  return (
    <div style={{ margin: '0 auto', padding: '0 20px', maxWidth: '95%' }}>
      <h1>Run ID: {runId}</h1>
      <h2 style={{ overflowWrap: 'anywhere' }}>{config['model_name']}</h2>
      <Config configData={config} />
      <div style={{ marginBottom: '10px' }}>
        <MetricChart 
          title="loss"
          data={chartData}
          metricName="loss"
          color1="#FFFF00"
          color2="#00FFFF"
        />
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 500px), 1fr))',
        gap: '10px',
      }}>
        {metricsList.map(metric => (
          <MetricChart
            title={metric}
            data={chartData}
            metricName={metric}
            color1="#FFFF00"
            color2="#00FFFF"
          />
        ))}
      </div>
    </div>
  );
}

export default RunDetails;