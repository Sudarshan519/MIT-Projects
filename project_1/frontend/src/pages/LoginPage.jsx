import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { ShieldAlert, Lock, User, Loader2 } from 'lucide-react';
import './LoginPage.css';

export default function LoginPage() {
  const [username, setUsername] = useState('analyst');
  const [password, setPassword] = useState('analyst123');
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsSubmitting(true);
    
    const result = await login(username, password);
    
    if (result.success) {
      navigate('/');
    } else {
      setError(result.error);
      setIsSubmitting(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-card glass-panel animate-fade-in">
        <div className="login-header">
          <div className="logo-circle">
            <ShieldAlert size={40} className="logo-icon-large" />
          </div>
          <h2>Sentinel Portal</h2>
          <p>Sign in to the Fraud Intelligence System</p>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {error && <div className="error-alert">{error}</div>}
          
          <div className="input-group">
            <label>Username</label>
            <div className="input-wrapper">
              <User size={18} className="input-icon" />
              <input
                type="text"
                className="input-field with-icon"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="input-group">
            <label>Password</label>
            <div className="input-wrapper">
              <Lock size={18} className="input-icon" />
              <input
                type="password"
                className="input-field with-icon"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <button type="submit" className="btn btn-primary login-btn" disabled={isSubmitting}>
            {isSubmitting ? <Loader2 className="spinner" size={20} /> : 'Authenticate'}
          </button>
        </form>

        <div className="demo-credentials">
          <p><strong>Demo Access</strong></p>
          <div className="flex-between">
            <span>Analyst: <code>analyst</code> / <code>analyst123</code></span>
          </div>
          <div className="flex-between" style={{ marginTop: '4px' }}>
            <span>Admin: <code>admin</code> / <code>admin123</code></span>
          </div>
        </div>
      </div>
    </div>
  );
}
