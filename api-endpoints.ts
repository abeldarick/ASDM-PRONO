// Types
interface PredictionResponse {
  homeScore: number;
  awayScore: number;
  over15Probability: number;
  confidence: number;
  features: Record<string, any>;
}

interface MatchData {
  id: string;
  homeTeam: string;
  awayTeam: string;
  date: Date;
  competition: string;
  stats?: Record<string, any>;
}

// Routes de l'API
const routes = {
  auth: {
    // Authentification et gestion des utilisateurs
    'POST /api/auth/register': {
      body: {
        email: string;
        password: string;
        name: string;
      },
      response: { token: string; userId: string }
    },
    'POST /api/auth/login': {
      body: {
        email: string;
        password: string;
      },
      response: { token: string; userId: string }
    },
  },
  
  predictions: {
    // Prédictions et analyses
    'GET /api/predictions/match/:matchId': {
      params: { matchId: string },
      response: PredictionResponse
    },
    'GET /api/predictions/date/:date': {
      params: { date: string },
      response: {
        matches: MatchData[];
        predictions: PredictionResponse[];
      }
    },
    'POST /api/predictions/analyze': {
      body: {
        homeTeam: string;
        awayTeam: string;
        date: string;
        competition: string;
      },
      response: PredictionResponse
    }
  },
  
  matches: {
    // Gestion des matchs
    'GET /api/matches': {
      query: {
        date?: string;
        competition?: string;
        team?: string;
        limit?: number;
        offset?: number;
      },
      response: {
        matches: MatchData[];
        total: number;
      }
    },
    'GET /api/matches/:matchId/stats': {
      params: { matchId: string },
      response: {
        matchStats: Record<string, any>;
        historicalStats: Record<string, any>;
      }
    }
  },
  
  admin: {
    // Routes administrateur
    'POST /api/admin/update-models': {
      body: {
        modelType: 'statistical' | 'deep-learning';
        parameters?: Record<string, any>;
      },
      response: {
        success: boolean;
        metrics: Record<string, number>;
      }
    },
    'GET /api/admin/metrics': {
      response: {
        predictionsCount: number;
        accuracy: number;
        userCount: number;
        systemHealth: Record<string, any>;
      }
    }
  }
};

// Middleware de sécurité
const securityMiddleware = {
  rateLimiter: {
    window: '15m',
    max: 100
  },
  auth: {
    required: [
      '/api/predictions/*',
      '/api/admin/*'
    ],
    optional: [
      '/api/matches/*'
    ]
  },
  cors: {
    origins: ['https://yourapp.com'],
    methods: ['GET', 'POST', 'PUT', 'DELETE']
  }
};
