import React from 'react';
import './Config.css';

function flattenConfig(data, prefix = '') {
  let result = {};
  for (let key in data) {
    if (typeof data[key] === 'object' && data[key] !== null && !Array.isArray(data[key])) {
      let nested = flattenConfig(data[key], `${prefix}${key} >> `);
      result = {...result, ...nested};
    } else {
      result[`${prefix}${key}`] = data[key];
    }
  }
  return result;
}

function Config({ configData }) {
  if (!configData) {
    return <div>Loading...</div>;
  }

  const flattenedConfig = flattenConfig(configData);

  return (
    <div className="config-container">
      {Object.entries(flattenedConfig).map(([key, value], idx) => {
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