import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { convertToLocalTimestamp } from '../utils';

import ConfigModal from './ConfigModal';

function TableRow({ run }) {
  const config_data = JSON.parse(run.config || '{}');
  const num_epochs = run['num_trained_epochs']

  const timestamp = convertToLocalTimestamp(run.started_at);

  // Check if the run uses a mini dataset
  const isMiniRun = run.validation_dataset_type === 'mini';

  // define the visibility state for the config
  const [isConfigVisible, setConfigVisible] = useState(false);

  const toggleConfigVisibility = () => {
    setConfigVisible(!isConfigVisible);
  };

  function formatParamCount(count) {
    if (count >= 1e6) {
      return (count / 1e6).toFixed(2) + 'M';
    } else if (count >= 1e3) {
      return (count / 1e3).toFixed(2) + 'K';
    } else {
      return count.toString();
    }
  }

  return (
    <>
      <tr className={isMiniRun ? 'mini-run' : ''}>
        <td>
          <Link to={`/run/${run.run_id}`}>{run.run_id}</Link>
        </td>
        <td>{timestamp}</td>
        {run.run_status === 'IN_PROGRESS' ? (
          <td style={{ animation: `in-progress-pulsing-fade 2s infinite linear` }}>
            {run.run_status}
            <span style={{ animation: `dot-one-animation 1s infinite linear` }}>.</span>
            <span style={{ animation: `dot-two-animation 1s infinite linear` }}>.</span>
            <span style={{ animation: `dot-three-animation 1s infinite linear` }}>.</span>
          </td>
        ) : (
          <td>{run.run_status}</td>
        )}
        <td>{num_epochs}</td>
        <td>{formatParamCount(run['model_parameter_count'])}</td>
        <td>{run['final_accuracy'].toFixed(2)}</td>
        <td>{run['final_top_5_accuracy'].toFixed(2)}</td>
        <td>{run['final_f1_score_macro'].toFixed(2)}</td>
        <td>
          <button className="btn btn-primary" onClick={toggleConfigVisibility}>View Config</button>
        </td>
      </tr>
      {isConfigVisible && 
        <ConfigModal configData={config_data} onClose={toggleConfigVisibility} />
      }
    </>
  );
}

export default TableRow;