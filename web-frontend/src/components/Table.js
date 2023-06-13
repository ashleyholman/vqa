import React from 'react';
import TableRow from './TableRow';

function Table({ data }) {
  return (
    <table className="table table-dark">
      <thead>
        <tr>
          <th>Run ID</th>
          <th>Timestamp</th>
          <th>Status</th>
          <th>Num Epochs</th>
          <th>Accuracy</th>
          <th>Top 5 Accuracy</th>
          <th>Config</th>
        </tr>
      </thead>
      <tbody>
        {data.map((item, idx) => (
          <TableRow key={idx} data={item} />
        ))}
      </tbody>
    </table>
  );
}

export default Table;