import React from 'react';
import './Config.css';

function Config({ configData }) {
  if (!configData) {
    return <div>Loading...</div>;
  }

  return (
    <div className="config-container">
      {Object.entries(configData).map(([key, value], idx) => {
        if (key === 'model_name') {
          return null;
        }
        let displayValue;
        if (typeof value === 'boolean') {
          displayValue = (
            <span className={value ? 'boolean-true' : 'boolean-false'}>
              {value.toString()}
            </span>
          );
        } else {
          displayValue = value;
        }

        return (
          <div key={idx} className="config-row">
            <div className="config-key">{key}</div>
            <div className="config-value">{displayValue}</div>
          </div>
        );
      })}
    </div>
  );
}

export default Config;