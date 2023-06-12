import React from 'react';

function Config({ configData }) {
  return (
    <div className="card card-body">
      <div className="row">
        {Object.entries(configData).map(([key, value], idx) => (
          <div key={idx} className="col-sm-6">
            <strong>{key}</strong>: {value}
          </div>
        ))}
      </div>
    </div>
  );
}

export default Config;
