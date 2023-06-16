import React from 'react';
import githubMark from '../svg/github-mark-white.svg';
import './Header.css';
import BebasNeueRegular from '../fonts/BebasNeue-Regular.ttf';

function Header() {
  return (
    <div className="header-container">
      <h1 className="header-title">Visual Question Answering</h1>
      <a href="https://github.com/ashleyholman/vqa" className="github-link">
        <img src={githubMark} alt="github" className="github-mark" />
      </a>
    </div>
  );
}

export default Header;