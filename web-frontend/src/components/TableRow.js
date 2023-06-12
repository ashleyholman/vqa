import React, { useState } from 'react';
import Config from './Config';

import { convertToLocalTimestamp } from '../utils';

function TableRow({ data }) {
  const [run, outfile, metrics] = data;

  // handle the case when metrics is empty
  const keys = Object.keys(metrics);
  const maxKey = keys.length > 0 ? Math.max(...keys.map(Number)) : null;
  const last_epoch_metrics = maxKey !== null ? metrics[maxKey] : {};

  const accuracy = last_epoch_metrics['validation_accuracy'];
  const top_5_accuracy = last_epoch_metrics['validation_top_5_accuracy'];
  const config_data = JSON.parse(run.config || '{}');
  const num_epochs = keys.length;

  const timestamp = convertToLocalTimestamp(run.started_at);

  // define the visibility state for the config
  const [isConfigVisible, setConfigVisible] = useState(false);

  const toggleConfigVisibility = () => {
    setConfigVisible(!isConfigVisible);
  };

  return (
    <>
      <tr>
        <td>{run.run_id}</td>
        <td>{timestamp}</td>
        <td>{run['run_status']}</td>
        <td>{num_epochs}</td>
        <td>{accuracy}</td>
        <td>{top_5_accuracy}</td>
        <td>
          <button class="btn btn-primary" onClick={toggleConfigVisibility}>Show/Hide Config</button>
        </td>
      </tr>
      {isConfigVisible && 
        <tr>
          <td colSpan="7">
            <Config configData={config_data} />
          </td>
        </tr>
      }
    </>
  );
}

export default TableRow;