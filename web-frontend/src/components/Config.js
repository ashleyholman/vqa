import React from 'react';

function Config({ configData }) {
  const containerStyle = {
    backgroundColor: '#242424',
    color: 'white',
    padding: '20px',
    borderRadius: '5px',
    fontFamily: 'Arial, sans-serif',
  };

  const rowStyle = {
    display: 'flex',
    justifyContent: 'space-between',
    borderBottom: '1px solid #444',
    padding: '10px 0',
    fontSize: '16px',
  };

  const keyStyle = {
    fontWeight: 'bold',
  };

  const valueStyle = {
    fontWeight: 'normal',
    textAlign: 'right',
  };

  return (
    <div style={containerStyle}>
      {Object.entries(configData).map(([key, value], idx) => {
        // skip model_name as it's too wide and not necessary to display.  we can display it elsewhere.
        if (key === 'model_name') {
          return null;
        }
        let displayValue;
        if (typeof value === 'boolean') {
          displayValue = (
            <span style={{color: value ? 'limegreen' : 'red'}}>
              {value.toString()}
            </span>
          );
        } else {
          displayValue = value;
        }

        return (
          <div key={idx} style={rowStyle}>
            <div style={keyStyle}>{key}</div>
            <div style={valueStyle}>{displayValue}</div>
          </div>
        );
      })}
    </div>
  );
}

export default Config;