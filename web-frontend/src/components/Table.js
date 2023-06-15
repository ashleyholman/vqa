import React, { useState, useEffect } from 'react';
import TableRow from './TableRow';
import { AiOutlineArrowUp, AiOutlineArrowDown } from 'react-icons/ai'; // import icons
import './Table.css';

function Table({ data }) {
  // Add state for sort field and order
  const [sortField, setSortField] = useState('started_at');
  const [sortOrder, setSortOrder] = useState('desc');

  // Add sortedData state to keep the sorted data
  const [sortedData, setSortedData] = useState([]);

  // Create effect that updates sortedData whenever data, sortField or sortOrder change
  useEffect(() => {
    const newData = [...data];
    newData.sort((a, b) => {
      if (a[sortField] < b[sortField]) {
        return sortOrder === 'asc' ? -1 : 1;
      }
      if (a[sortField] > b[sortField]) {
        return sortOrder === 'asc' ? 1 : -1;
      }
      return 0;
    });
    setSortedData(newData);
  }, [data, sortField, sortOrder]);

  // Function to handle clicking a column header
  const handleSort = (field) => {
    if (field === sortField) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortOrder('desc');
    }
  };

  // Function to render the sort arrow icon
  const renderSortArrow = (field) => {
    if (sortField === field) {
      return sortOrder === 'asc' ? <AiOutlineArrowUp /> : <AiOutlineArrowDown />;
    }
    return <AiOutlineArrowDown className="hidden-icon" />;
  };

  return (
    <div className="table-responsive">
      <table className="table table-dark">
      <thead>
          <tr>
            <th onClick={() => handleSort('run_id')}><div className="header-div">Run ID {renderSortArrow('run_id')}</div></th>
            <th onClick={() => handleSort('started_at')}><div className="header-div">Timestamp {renderSortArrow('started_at')}</div></th>
            <th onClick={() => handleSort('run_status')}><div className="header-div">Status {renderSortArrow('run_status')}</div></th>
            <th onClick={() => handleSort('num_trained_epochs')}><div className="header-div">Num Epochs {renderSortArrow('num_trained_epochs')}</div></th>
            <th onClick={() => handleSort('final_accuracy')}><div className="header-div">Accuracy {renderSortArrow('final_accuracy')}</div></th>
            <th onClick={() => handleSort('final_top_5_accuracy')}><div className="header-div">Top 5 Accuracy {renderSortArrow('final_top_5_accuracy')}</div></th>
            <th onClick={() => handleSort('final_f1_score_macro')}><div className="header-div">F1 Macro {renderSortArrow('final_f1_score_macro')}</div></th>
            <th>Config</th>
          </tr>
        </thead>
        <tbody>
          {sortedData.map((item, idx) => (
            <TableRow key={idx} run={item} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Table;