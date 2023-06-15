import React from 'react';
import Config from './Config';
import './ConfigModal.css';

function ConfigModal({ configData, onClose }) {
  return (
    <div className="config-modal" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <button className="close-button" onClick={onClose}>X</button>
        <div className="modal-config-container">
          <Config configData={configData} />
        </div>
      </div>
    </div>
  );
}

export default ConfigModal;