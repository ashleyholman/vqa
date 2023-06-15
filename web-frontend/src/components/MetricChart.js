import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, Label } from 'recharts';

import './MetricChart.css';

function MetricChart({ title, data, metricName, color1, color2 }) {
  const metric1Name = `training_${metricName}`;
  const metric2Name = `validation_${metricName}`;

  const allValues = data.flatMap(d => [d[metric1Name], d[metric2Name]]);
  const minVal = Math.floor(Math.min(...allValues));
  const maxVal = Math.ceil(Math.max(...allValues));
  const yticks = Array.from({ length: maxVal - minVal + 1 }, (_, i) => minVal + i);
  const domain = [minVal, maxVal];

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="custom-tooltip" style={{backgroundColor: '#333', borderColor: '#999', padding: '10px', borderRadius: '5px'}}>
          <p className="label"><strong>Epoch {label}</strong></p>
          {payload.map((entry, index) => (
            <p key={`item-${index}`} style={{color: entry.color}}>
              {`${entry.name} : ${entry.value}`}
            </p>
          ))}
        </div>
      );
    }
  
    return null;
  };

  return (
    <div className="metric-chart-container">
      <div className="metric-chart-inner">
          <h3 className="metric-chart-title">{title}</h3>
          <ResponsiveContainer width='100%' height={400}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 20 }}>
              <Line type="monotone" dataKey={metric1Name} stroke={color1} strokeWidth={2} dot={<circle r={3} fill={color1} />} />
              <Line type="monotone" dataKey={metric2Name} stroke={color2} strokeWidth={2} dot={<circle r={3} fill={color2} />} />
              <CartesianGrid stroke="#ccc" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="epoch" stroke="white">
                <Label value="Epochs" offset={-10} position="insideBottom" className="label-style" />
              </XAxis>
              <YAxis domain={domain} ticks={yticks} stroke="white">
                <Label angle={-90} value={metricName} position="insideLeft" className="label-style" />
              </YAxis>
              <Tooltip content={<CustomTooltip />} contentStyle={{ backgroundColor: '#333', borderColor: '#999' }} />
              <Legend verticalAlign="top" wrapperStyle={{ paddingBottom: '20px' }} />
            </LineChart>
          </ResponsiveContainer>
      </div>
    </div>
  );
}

export default MetricChart;