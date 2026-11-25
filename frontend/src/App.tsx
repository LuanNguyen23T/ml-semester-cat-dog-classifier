import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import { BarChart3, Cat, History, Zap, Moon, Sun } from 'lucide-react';
import { HomePage } from './pages/Home';
import { HistoryPage } from './pages/History';
import { StatisticsPage } from './pages/Statistics';
import { PredictionSessionProvider } from './contexts/PredictionSessionContext';
import './index.css';

function NavBar({ isDark, toggleTheme }: { isDark: boolean, toggleTheme: () => void }) {
  const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    padding: '8px 16px',
    borderRadius: 10,
    fontSize: 14,
    fontWeight: 600,
    textDecoration: 'none',
    transition: 'all 0.2s ease',
    color: isActive ? 'var(--color-accent)' : 'var(--color-text-muted)',
    background: isActive ? 'rgba(139,92,246,0.12)' : 'transparent',
    border: isActive ? '1px solid rgba(139,92,246,0.25)' : '1px solid transparent',
  });

  return (
    <nav style={{
      position: 'sticky',
      top: 0,
      zIndex: 100,
      background: 'var(--color-nav-bg)',
      backdropFilter: 'blur(20px)',
      borderBottom: '1px solid var(--color-border)',
    }}>
      <div className="app-nav-inner" style={{
        maxWidth: 1100,
        margin: '0 auto',
        padding: '0 24px',
        display: 'flex',
        alignItems: 'center',
        height: 60,
        gap: 8,
      }}>
        {/* Logo */}
        <div className="app-brand" style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          marginRight: 24,
        }}>
          <div style={{
            width: 32,
            height: 32,
            borderRadius: 8,
            background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 16px rgba(139,92,246,0.4)',
          }}>
            <Cat size={18} color="white" />
          </div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 800, lineHeight: 1, letterSpacing: '-0.02em' }}>
              <span style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
              }}>
                Pet Breed
              </span>
              {' '}Classifier
            </div>
            <div className="app-brand-subtitle" style={{ fontSize: 10, color: 'var(--color-text-faint)', lineHeight: 1, marginTop: 2 }}>
              37 giống chó và mèo
            </div>
          </div>
        </div>

        {/* Nav links */}
        <NavLink to="/" style={navLinkStyle} end id="nav-home">
          <Zap size={15} />
          <span className="app-nav-label">Classify</span>
        </NavLink>

        <NavLink to="/history" style={navLinkStyle} id="nav-history">
          <History size={15} />
          <span className="app-nav-label">History</span>
        </NavLink>

        <NavLink to="/statistics" style={navLinkStyle} id="nav-statistics">
          <BarChart3 size={15} />
          <span className="app-nav-label">Statistics</span>
        </NavLink>
        
        {/* Theme Toggle */}
        <button 
          onClick={toggleTheme}
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 36,
            height: 36,
            borderRadius: '50%',
            border: '1px solid var(--color-border)',
            background: 'transparent',
            color: 'var(--color-text-muted)',
            cursor: 'pointer',
            marginLeft: 'auto',
            transition: 'all 0.2s ease',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = 'rgba(139,92,246,0.1)';
            e.currentTarget.style.color = 'var(--color-text)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = 'transparent';
            e.currentTarget.style.color = 'var(--color-text-muted)';
          }}
          aria-label="Toggle theme"
        >
          {isDark ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </nav>
  );
}

function App() {
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    if (isDark) {
      document.documentElement.setAttribute('data-theme', 'dark');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
  }, [isDark]);

  const toggleTheme = () => setIsDark(!isDark);

  return (
    <BrowserRouter>
      <PredictionSessionProvider>
        <NavBar isDark={isDark} toggleTheme={toggleTheme} />
        <main>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/statistics" element={<StatisticsPage />} />
          </Routes>
        </main>
      </PredictionSessionProvider>
    </BrowserRouter>
  );
}

export default App;
