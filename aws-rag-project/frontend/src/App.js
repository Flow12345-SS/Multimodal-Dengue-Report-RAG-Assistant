import React, { useState } from 'react';
import Dashboard from './components/Dashboard';

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div style={{ fontFamily: 'Arial, sans-serif', margin: '0 auto', maxWidth: '1200px', padding: '20px' }}>
      <header style={{ borderBottom: '2px solid #007bff', paddingBottom: '10px', marginBottom: '20px', display: 'flex', justifyContent: 'space-between' }}>
        <h1 style={{ color: '#007bff', margin: 0 }}>🦟 Dengue Assistant (AWS Bedrock)</h1>
        <nav>
          <button onClick={() => setActiveTab('dashboard')} style={navButtonStyle(activeTab === 'dashboard')}>Dashboard</button>
        </nav>
      </header>
      
      <main>
        {activeTab === 'dashboard' && <Dashboard />}
      </main>
    </div>
  );
}

const navButtonStyle = (isActive) => ({
  background: isActive ? '#007bff' : 'transparent',
  color: isActive ? 'white' : '#007bff',
  border: '1px solid #007bff',
  padding: '8px 16px',
  borderRadius: '4px',
  cursor: 'pointer',
  marginLeft: '10px'
});

export default App;
