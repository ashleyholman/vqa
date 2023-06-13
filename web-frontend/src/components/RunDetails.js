import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Config from './Config';
import MetricChart from './MetricChart';

function RunDetails() {
  const { runId } = useParams();
  const [runData, setRunData] = useState(null);

  // This structure defines the layout of graphs on the page after the initial 'loss' graph
  const metricPairs = [
    ['accuracy', 'top_5_accuracy'],
    ['precision_macro', 'precision_micro'],
    ['recall_macro', 'recall_micro'],
    ['f1_score_macro', 'f1_score_micro']
  ];

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
    <div style={{ margin: '0 auto', maxWidth: '95%' }}>
      <h1>Run ID: {runId}</h1>
      <h2>{config['model_name']}</h2>
      <Config configData={config} />
      <MetricChart 
        title="loss"
        data={chartData}
        metricName="loss"
        color1="#FFFF00"
        color2="#00FFFF"
      />
      {metricPairs.map(([metric1, metric2]) => (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'stretch' }}>
          <div style={{ flex: 1, flexBasis: 0, overflow: 'hidden' }}>
            <MetricChart
                title={metric1}
                data={chartData}
                metricName={metric1}
                color1="#FFFF00"
                color2="#00FFFF"
            />
          </div>
          <div style={{ flex: 1, flexBasis: 0, overflow: 'hidden' }}>
            <MetricChart
                title={metric2}
                data={chartData}
                metricName={metric2}
                color1="#FFFF00"
                color2="#00FFFF"
            />
          </div>
        </div>
      ))}
    </div>
  );
}

export default RunDetails;