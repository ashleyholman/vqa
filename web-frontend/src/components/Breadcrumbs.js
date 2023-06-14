import { useLocation } from 'react-router-dom';
import { Link } from 'react-router-dom';

function Breadcrumbs() {
  const location = useLocation();
  let breadcrumbs = null;
  
  if (location.pathname === '/') {
    breadcrumbs = (
      <Link to="/" style={{ color: '#00A878', fontWeight: 'bold' }}>Runs</Link>
    );
  } else if (location.pathname.startsWith('/run')) {
    const runId = location.pathname.split('/')[2];
    breadcrumbs = (
      <>
        <Link to="/" style={{ color: '#00A878', fontWeight: 'bold' }}>Runs</Link> -> {runId}
      </>
    );
  }
  
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'flex-start',
      alignItems: 'center',
      padding: '5px 10px',
      backgroundColor: '#333'
    }}>
      <div style={{ color: 'white', lineHeight: '1.2' }}>{breadcrumbs}</div>
    </div>
  );
}

export default Breadcrumbs;