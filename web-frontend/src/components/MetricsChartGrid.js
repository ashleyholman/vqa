import React from 'react';
import MetricChart from './MetricChart';

function MetricsChartGrid({ metrics }) {
  const metricsList = ['accuracy', 'top_5_accuracy', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'f1_score_macro', 'f1_score_micro'];

  const chartData = Object.keys(metrics).map((key) => ({
    epoch: metrics[key].epoch,
    ...metrics[key] // Spread remaining properties
  }));

  return (
    <div>
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

export default MetricsChartGrid;