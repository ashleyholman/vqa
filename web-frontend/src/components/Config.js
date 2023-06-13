import React from 'react';

function Config({ configData }) {
  return (
    <div className="card bg-dark text-white card-body">
      {Object.entries(configData).map(([key, value], idx) => {

        // Special rendering for boolean values
        let value_class = '';
        if (typeof value === 'boolean') {
          value_class = "badge " + (value ? 'badge-success' : 'badge-danger');
          value = value.toString();
        }

        return (
          <div key={idx} className="row mb-2">
            <div className="col-sm-3">
              <span className="badge badge-info">{key}</span>
            </div>
            <div className="col-sm-9">
              <span className={value_class}>{value}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default Config;