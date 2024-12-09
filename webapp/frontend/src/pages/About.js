import { Container, Typography, Grid, IconButton, Box } from '@mui/material';
import { motion } from 'framer-motion';
import { 
  Visibility, Speed, Security, 
  Timeline, Psychology, CloudDone,
  GitHub, LinkedIn, Twitter,
  Rocket as RocketIcon,
  PeopleAlt as TeamIcon,
  Lightbulb as VisionIcon,
} from '@mui/icons-material';
import { GlassPaper } from '../components/StyledComponents';

const FeatureCard = motion(GlassPaper);

function About() {
  const features = [
    {
      icon: Visibility,
      title: 'Advanced CV Technology',
      description: 'State-of-the-art computer vision algorithms for satellite imagery analysis',
      color: '#0984E3',
    },
    {
      icon: Speed,
      title: 'Real-time Processing',
      description: 'Fast and efficient processing of large-scale satellite data',
      color: '#00B894',
    },
    {
      icon: Security,
      title: 'Secure & Reliable',
      description: 'Enterprise-grade security with reliable infrastructure',
      color: '#6C5CE7',
    }
  ];

  const values = [
    {
      icon: RocketIcon,
      title: 'Innovation',
      description: 'Pushing boundaries in satellite imagery analysis',
    },
    {
      icon: TeamIcon,
      title: 'Collaboration',
      description: 'Working together to achieve excellence',
    },
    {
      icon: VisionIcon,
      title: 'Vision',
      description: 'Shaping the future of urban accessibility',
    },
  ];

  const timeline = [
    {
      year: '2023',
      title: 'Platform Launch',
      description: 'Initial release of SightLink platform',
      icon: CloudDone,
    },
    {
      year: '2024',
      title: 'Advanced Features',
      description: 'Introduction of advanced analysis capabilities',
      icon: Psychology,
    },
    {
      year: 'Future',
      title: 'Global Expansion',
      description: 'Expanding our services worldwide',
      icon: Timeline,
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 8 }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {/* Hero Section */}
          <Box sx={{ textAlign: 'center', mb: 10 }}>
            <Typography 
              variant="h3" 
              gutterBottom
              sx={{
                fontWeight: 700,
                fontSize: '2.5rem',
                background: 'linear-gradient(45deg, #2D3436 30%, #636E72 90%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              About SightLink
            </Typography>
            <Typography 
              variant="h6"
              color="text.secondary"
              sx={{ 
                maxWidth: '700px',
                mx: 'auto',
                mb: 3,
                fontWeight: 500,
                lineHeight: 1.6,
                fontSize: '1.125rem',
              }}
            >
              Transforming satellite imagery analysis with advanced computer vision
            </Typography>
          </Box>

          {/* Features Section */}
          <Grid container spacing={4} sx={{ mb: 10 }}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <FeatureCard
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ 
                    translateY: -8,
                    boxShadow: '0 12px 40px rgba(0, 0, 0, 0.12)'
                  }}
                  sx={{ textAlign: 'center', p: 4 }}
                >
                  <Box
                    sx={{
                      width: 64,
                      height: 64,
                      borderRadius: '20px',
                      background: `linear-gradient(135deg, ${feature.color} 0%, ${feature.color}80 100%)`,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      mb: 3,
                      mx: 'auto',
                      boxShadow: `0 8px 24px ${feature.color}40`,
                    }}
                  >
                    <feature.icon sx={{ fontSize: 30, color: 'white' }} />
                  </Box>
                  <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                    {feature.title}
                  </Typography>
                  <Typography color="text.secondary" sx={{ lineHeight: 1.7 }}>
                    {feature.description}
                  </Typography>
                </FeatureCard>
              </Grid>
            ))}
          </Grid>

          {/* Mission Section */}
          <GlassPaper 
            sx={{ 
              p: 6, 
              mb: 10, 
              textAlign: 'center',
              background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%)',
            }}
          >
            <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
              Our Mission
            </Typography>
            <Typography 
              variant="body1" 
              color="text.secondary" 
              sx={{ 
                maxWidth: '800px', 
                mx: 'auto',
                lineHeight: 1.8,
                fontSize: '1.1rem'
              }}
            >
              SightLink is dedicated to revolutionizing the way we analyze and understand satellite imagery. 
              Our platform combines cutting-edge computer vision technology with user-friendly interfaces 
              to make satellite data analysis accessible and efficient for everyone.
            </Typography>
          </GlassPaper>

          {/* Values Section */}
          <Typography 
            variant="h4" 
            gutterBottom 
            sx={{ 
              textAlign: 'center', 
              mb: 4,
              fontWeight: 600,
            }}
          >
            Our Values
          </Typography>
          <Grid container spacing={4} sx={{ mb: 10 }}>
            {values.map((value, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <GlassPaper sx={{ 
                    textAlign: 'center',
                    p: 4,
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <Box
                      sx={{
                        width: 56,
                        height: 56,
                        borderRadius: '16px',
                        backgroundColor: 'rgba(9, 132, 227, 0.08)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 2,
                      }}
                    >
                      <value.icon sx={{ fontSize: 28, color: '#0984E3' }} />
                    </Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      {value.title}
                    </Typography>
                    <Typography color="text.secondary">
                      {value.description}
                    </Typography>
                  </GlassPaper>
                </motion.div>
              </Grid>
            ))}
          </Grid>

          {/* Timeline Section */}
          <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4, fontWeight: 600 }}>
            Our Journey
          </Typography>
          <Grid container spacing={4} sx={{ mb: 10 }}>
            {timeline.map((item, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <GlassPaper sx={{ 
                    textAlign: 'center',
                    p: 4,
                    position: 'relative',
                    overflow: 'hidden',
                  }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        color: 'primary.main',
                        fontWeight: 600,
                        mb: 2,
                      }}
                    >
                      {item.year}
                    </Typography>
                    <Box
                      sx={{
                        width: 56,
                        height: 56,
                        borderRadius: '16px',
                        background: 'linear-gradient(135deg, rgba(9, 132, 227, 0.1) 0%, rgba(116, 185, 255, 0.1) 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        mb: 2,
                        mx: 'auto',
                      }}
                    >
                      <item.icon sx={{ fontSize: 28, color: '#0984E3' }} />
                    </Box>
                    <Typography variant="h6" gutterBottom sx={{ fontWeight: 600 }}>
                      {item.title}
                    </Typography>
                    <Typography color="text.secondary">
                      {item.description}
                    </Typography>
                  </GlassPaper>
                </motion.div>
              </Grid>
            ))}
          </Grid>

        </motion.div>
      </Box>
    </Container>
  );
}

export default About; 