import { useParams } from 'react-router-dom';

function RunDetails() {
  const { runId } = useParams(); // get runId from URL parameters

  // Then, you can use runId in your component or fetch data based on this id

  return (
    <h1>Run ID: {runId}</h1>
  );
}

export default RunDetails;