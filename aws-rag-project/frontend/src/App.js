import React, { useState } from 'react';
import Dashboard from './components/Dashboard';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div style={containerStyle}>
      {/* SaaS Hero Banner (Dark Navy #050814 + Neon Accent #84CC16) */}
      <header style={headerBannerStyle}>
        <div style={taglineBadgeStyle}>
          <span style={pulseDotStyle}></span>
          <span>AI-POWERED CLINICAL DECISION SUPPORT PLATFORM</span>
          <span style={{ opacity: 0.4 }}>|</span>
          <span style={{ color: '#D9F99D' }}>AWS BEDROCK RAG</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '15px' }}>
          <div>
            <h1 style={titleStyle}>
              🩺 Multimodal Dengue Report <span style={neonHighlightStyle}>RAG Assistant</span>
            </h1>
            <p style={subtitleStyle}>
              Clinical Decision Support System • High-Precision Diagnostics & Grounded Evidence Synthesis
            </p>
          </div>
          <nav style={{ display: 'flex', gap: '10px' }}>
            <button 
              onClick={() => setActiveTab('dashboard')} 
              style={navButtonStyle(activeTab === 'dashboard')}
            >
              Dashboard
            </button>
          </nav>
        </div>
        <div style={heroChipRowStyle}>
          <span style={heroChipStyle}>🟢 Bedrock Knowledge Base Active</span>
          <span style={heroChipStyle}>⚡ Real-Time Multimodal Retrieval</span>
          <span style={heroChipStyle}>🛡️ Clinical Guardrails Synced</span>
        </div>
      </header>
      
      <main>
        {activeTab === 'dashboard' && <Dashboard />}
      </main>
    </div>
  );
}

const containerStyle = {
  fontFamily: "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif",
  margin: '0 auto',
  maxWidth: '1280px',
  padding: '24px 20px',
  backgroundColor: '#74D116',
  minHeight: '100vh',
  boxSizing: 'border-box'
};

const headerBannerStyle = {
  background: '#0D0D0D',
  border: '2px solid #74D116',
  borderRadius: '22px',
  padding: '28px 32px',
  marginBottom: '28px',
  boxShadow: '0 20px 50px -10px rgba(0, 0, 0, 0.5), 0 0 35px rgba(116, 209, 22, 0.3)',
  color: '#FFFFFF'
};

const taglineBadgeStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '8px',
  background: 'rgba(132, 204, 22, 0.15)',
  border: '1.5px solid rgba(132, 204, 22, 0.55)',
  color: '#84CC16',
  padding: '6px 14px',
  borderRadius: '9999px',
  fontSize: '0.8rem',
  fontWeight: '800',
  letterSpacing: '0.08em',
  textTransform: 'uppercase',
  marginBottom: '14px'
};

const pulseDotStyle = {
  width: '8px',
  height: '8px',
  backgroundColor: '#84CC16',
  borderRadius: '50%',
  boxShadow: '0 0 10px #84CC16',
  display: 'inline-block'
};

const titleStyle = {
  fontSize: '2rem',
  fontWeight: '900',
  color: '#FFFFFF',
  margin: '0 0 6px 0',
  letterSpacing: '-0.02em'
};

const neonHighlightStyle = {
  color: '#84CC16',
  textShadow: '0 0 20px rgba(132, 204, 22, 0.95)'
};

const subtitleStyle = {
  fontSize: '0.98rem',
  color: '#D9F99D',
  margin: 0,
  fontWeight: '600',
  opacity: 0.95
};

const heroChipRowStyle = {
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  flexWrap: 'wrap',
  marginTop: '16px'
};

const heroChipStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '6px',
  background: 'rgba(5, 8, 20, 0.85)',
  border: '1.5px solid rgba(132, 204, 22, 0.4)',
  color: '#F8FAFC',
  fontSize: '0.8rem',
  fontWeight: '700',
  padding: '4px 12px',
  borderRadius: '9999px'
};

const navButtonStyle = (isActive) => ({
  background: isActive ? 'linear-gradient(135deg, #84CC16 0%, #65A30D 100%)' : '#050814',
  color: '#FFFFFF',
  border: isActive ? 'none' : '1.5px solid #84CC16',
  padding: '10px 22px',
  borderRadius: '12px',
  fontWeight: '800',
  fontSize: '0.92rem',
  cursor: 'pointer',
  boxShadow: isActive ? '0 6px 20px rgba(132, 204, 22, 0.45)' : 'none',
  transition: 'all 0.25s ease'
});

export default App;
