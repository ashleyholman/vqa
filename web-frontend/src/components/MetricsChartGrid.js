import React from 'react';
import MetricChart from './MetricChart';
import './MetricsChartGrid.css';

function MetricsChartGrid({ metrics, isMiniRun }) {
  const metricsList = ['accuracy', 'top_5_accuracy', 'precision_macro', 'precision_micro', 'recall_macro', 'recall_micro', 'f1_score_macro', 'f1_score_micro'];

  const chartData = Object.keys(metrics).map((key) => ({
    epoch: metrics[key].epoch,
    ...metrics[key] // Spread remaining properties
  }));

  return (
    <div>
      <div className="metrics-chart">
        <MetricChart 
          title="loss"
          data={chartData}
          metricName="loss"
          isMiniRun={isMiniRun}
        />
      </div>
      <div className="metrics-chart-grid">
        {metricsList.map((metric, index) => (
          <MetricChart
            key={index}
            title={metric}
            data={chartData}
            metricName={metric}
            isMiniRun={isMiniRun}
          />
        ))}
      </div>
    </div>
  );
}

export default MetricsChartGrid;