import { Container, Typography, Grid, IconButton, Box } from '@mui/material';
import { motion } from 'framer-motion';
import { 
  Visibility, Speed, Security, 
  Timeline, Psychology, CloudDone,
  GitHub, LinkedIn, Twitter
} from '@mui/icons-material';
import { 
  GlassPaper, GradientTypography, IconBox, 
  PageContainer, HeroSection 
} from '../components/StyledComponents';

function About() {
  const features = [
    {
      icon: Visibility,
      title: 'Advanced CV Technology',
      description: 'State-of-the-art computer vision algorithms for satellite imagery analysis',
    },
    {
      icon: Speed,
      title: 'Real-time Processing',
      description: 'Fast and efficient processing of large-scale satellite data',
    },
    {
      icon: Security,
      title: 'Secure & Reliable',
      description: 'Enterprise-grade security with reliable infrastructure',
    }
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

  const fadeInUp = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.5 }
  };

  return (
    <Container maxWidth="lg">
      <PageContainer>
        <motion.div {...fadeInUp}>
          <Box sx={{ textAlign: 'center', mb: 6 }}>
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
                mb: 4,
                fontWeight: 500,
                lineHeight: 1.6,
                fontSize: '1.125rem',
              }}
            >
              Transforming satellite imagery analysis with advanced computer vision
            </Typography>
          </Box>

          <Grid container spacing={4} sx={{ mb: 8 }}>
            {features.map((feature, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  {...fadeInUp}
                  transition={{ delay: index * 0.1 }}
                >
                  <GlassPaper sx={{ 
                    textAlign: 'center',
                    '&:hover': {
                      transform: 'translateY(-4px)',
                    }
                  }}>
                    <IconBox>
                      <feature.icon sx={{ fontSize: 30 }} />
                    </IconBox>
                    <Typography variant="h5" gutterBottom sx={{ fontWeight: 600 }}>
                      {feature.title}
                    </Typography>
                    <Typography color="text.secondary">
                      {feature.description}
                    </Typography>
                  </GlassPaper>
                </motion.div>
              </Grid>
            ))}
          </Grid>

          <GlassPaper sx={{ 
            p: 6, 
            mb: 8, 
            textAlign: 'center',
            background: 'linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(255, 255, 255, 0.85) 100%)',
          }}>
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

          <Typography variant="h4" gutterBottom sx={{ textAlign: 'center', mb: 4 }}>
            Our Journey
          </Typography>
          <Grid container spacing={4} sx={{ mb: 8 }}>
            {timeline.map((item, index) => (
              <Grid item xs={12} md={4} key={index}>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <GlassPaper sx={{ textAlign: 'center' }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        color: 'primary.main',
                        fontWeight: 600,
                        mb: 2 
                      }}
                    >
                      {item.year}
                    </Typography>
                    <item.icon sx={{ fontSize: 40, color: 'secondary.main', mb: 2 }} />
                    <Typography variant="h6" gutterBottom>
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

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="h6" gutterBottom>
              Connect With Us
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center', gap: 2 }}>
              {[GitHub, LinkedIn, Twitter].map((Icon, index) => (
                <IconButton
                  key={index}
                  sx={{
                    color: 'text.secondary',
                    '&:hover': {
                      color: 'primary.main',
                    }
                  }}
                >
                  <Icon />
                </IconButton>
              ))}
            </Box>
          </Box>
        </motion.div>
      </PageContainer>
    </Container>
  );
}

export default About; 