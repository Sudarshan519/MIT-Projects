import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { ShieldAlert, LayoutDashboard, List, UploadCloud, Calculator, LogOut } from 'lucide-react';
import clsx from 'clsx';
import './Layout.css';

export default function Layout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { to: '/alerts', icon: List, label: 'Alert Queue' },
    { to: '/score', icon: UploadCloud, label: 'Score Transactions' },
    { to: '/cost-analysis', icon: Calculator, label: 'Cost Analysis' },
  ];

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-header">
          <ShieldAlert className="logo-icon" size={32} />
          <div>
            <h1 className="logo-text">Sentinel</h1>
            <p className="logo-subtext">Fraud Intelligence</p>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => clsx('nav-link', isActive && 'active')}
            >
              <item.icon size={20} />
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="user-info">
            <div className="avatar">{user?.full_name.charAt(0)}</div>
            <div>
              <p className="user-name">{user?.full_name}</p>
              <p className="user-role">{user?.role}</p>
            </div>
          </div>
          <button onClick={handleLogout} className="logout-btn" title="Logout">
            <LogOut size={18} />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <Outlet />
      </main>
    </div>
  );
}
