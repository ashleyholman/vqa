import React from 'react';
import Config from './Config';

function ConfigModal({ configData, onClose }) {
  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100vw',
        height: '100vh',
        backgroundColor: 'rgba(0,0,0,0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#333',
          padding: '5px',
          width: '90%',
          maxWidth: '800px',
          position: 'relative',
          border: '1px solid #666',
          overflow: 'hidden', // hide the overflow
        }}
        onClick={e => e.stopPropagation()}
      >
        <button
          style={{
            position: 'absolute',
            right: '0.2em',
            top: '0.2em',
            background: '#aaa',
            color: 'black',
            border: '1px solid #666',
            width: '20px',
            height: '20px',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            boxSizing: 'border-box',
            cursor: 'pointer',
            zIndex: 1,
          }}
          onClick={onClose}
        >
          X
        </button>
        <div
          style={{
            maxHeight: '80vh', // set maximum height
            overflowY: 'auto', // make it scrollable
            padding: '20px', // add some space around the content
          }}
        >
          <Config configData={configData} />
        </div>
      </div>
    </div>
  );
}

export default ConfigModal;