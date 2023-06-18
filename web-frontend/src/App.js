import React, { useState, useEffect } from 'react';
import { Routes, Route } from 'react-router-dom';
import { ErrorAnalysisProvider } from './contexts/ErrorAnalysisContext';

import Breadcrumbs from './components/Breadcrumbs';
import Header from './components/Header';
import Table from './components/Table';
import RunDetails from './components/RunDetails';

import './App.css';

function App() {
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // fetch data here
    fetch(process.env.PUBLIC_URL + '/data/runs.json')
      .then((response) => response.text())
      .then((text) => {
        try {
          const jsonData = JSON.parse(text);
          setData(jsonData); // Set the data here
          setIsLoading(false); // set loading state to false after data is fetched
        } catch (err) {
          console.error("There was an error parsing the JSON: ", err, "Raw content:", text);
        }
      })
      .catch((error) => console.error("Error fetching the data:", error));
  }, []);

  if (isLoading) {
    return <div>Loading...</div>; // render loading state if data has not yet been fetched
  }

  return (
    <div className="App">
      <ErrorAnalysisProvider>
        <Header />
        <Breadcrumbs />
        <Routes>
          <Route path="/" element={<Table data={data} />} />
          <Route path="/run/:runId/*" element={<RunDetails />} />
        </Routes>
      </ErrorAnalysisProvider>
    </div>
  );
}

export default App;