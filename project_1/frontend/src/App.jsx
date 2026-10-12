import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Layout from './components/Layout/Layout';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import AlertsPage from './pages/AlertsPage';
import UploadPage from './pages/UploadPage';
import CostAnalysisPage from './pages/CostAnalysisPage';

// Simple PrivateRoute wrapper
const PrivateRoute = ({ children }) => {
  const { user } = useAuth();
  return user ? children : <Navigate to="/login" replace />;
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        
        <Route path="/" element={<PrivateRoute><Layout /></PrivateRoute>}>
          <Route index element={<DashboardPage />} />
          <Route path="cost-analysis" element={<CostAnalysisPage />} />
          <Route path="alerts" element={<AlertsPage />} />
          <Route path="score" element={<UploadPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
