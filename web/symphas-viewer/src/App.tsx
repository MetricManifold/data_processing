import React, { useState } from 'react';
import DirectoryList from './DirectoryList';
import DataDisplay from './DataDisplay';
import { Container, Grid2, Paper } from '@mui/material';
import './App.css';

const App: React.FC = () => {
    const [selectedDirectory, setSelectedDirectory] = useState<string>("");

    return (
        <Container>
            <Grid2 container spacing={3}>
                <Grid2 size={3}>
                    <Paper elevation={3} style={{ padding: '16px' }}>
                        <DirectoryList onSelect={setSelectedDirectory} />
                    </Paper>
                </Grid2>
                <Grid2 size={9}>
                    <Paper elevation={3} style={{ padding: '16px' }}>
                        {selectedDirectory && <DataDisplay directory={selectedDirectory} />}
                    </Paper>
                </Grid2>
            </Grid2>
        </Container>
    );
}

export default App;