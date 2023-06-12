import React, { useState, useEffect } from 'react';
import './App.css';
import Table from './components/Table';

function App() {
  const [data, setData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // fetch data here
    fetch(process.env.PUBLIC_URL + '/processed_runs.json')
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
      <Table data={data} />
    </div>
  );
}

export default App;