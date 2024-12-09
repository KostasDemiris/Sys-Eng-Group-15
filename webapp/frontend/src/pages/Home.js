import { Container, Typography, Box } from '@mui/material';

function Home() {
  return (
    <Container>
      <Box sx={{ mt: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Welcome to Your App
        </Typography>
        <Typography variant="body1">
          This is the home page of your application.
        </Typography>
      </Box>
    </Container>
  );
}

export default Home; 