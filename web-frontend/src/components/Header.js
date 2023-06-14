import githubMark from '../svg/github-mark-white.svg';

function Header() {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '10px',
      background: 'linear-gradient(90deg, rgba(58,123,213,1) 0%, rgba(58,96,115,1) 50%, rgba(58,123,213,1) 100%)'
    }}>
      <h1 style={{ color: 'white' }}>Visual Question Answering</h1>
      <a href="https://github.com/ashleyholman/vqa">
        <img src={githubMark} alt="github" style={{ width: '30px', height: '30px' }} />
      </a>
    </div>
  );
}

export default Header;
