import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { API_ENDPOINT } from './config';
import { TextField, Button, List, Typography, Container, ListItemButton, ListItemText } from '@mui/material';

function DirectoryList({ onSelect }: { onSelect: (dir: string) => void }) {
    const [directories, setDirectories] = useState<string[]>([]);
    const [currentDirectory, setCurrentDirectory] = useState<string>("");

    const handleGetDirectories = () => {
        axios.get(`${API_ENDPOINT}/list_directory`)
            .then(response => setDirectories(response.data))
            .catch(error => console.error('Error fetching directories:', error));
    }

    const handleUpdateDirectory = () => {
        axios.post(`${API_ENDPOINT}/set_directory`, { path: currentDirectory })
            .then(() => handleGetDirectories())
            .catch(error => console.error('Error setting directory:', error));
    };

    useEffect(() => {
        handleGetDirectories();
    }, [currentDirectory]);

    return (
        <Container>
            <Typography variant="h4" gutterBottom>
                Directories
            </Typography>
            <TextField
                label="Enter directory name"
                variant="standard"
                value={currentDirectory}
                onChange={(e) => setCurrentDirectory(e.target.value)}
                fullWidth
                margin="normal"
                size="small"
            />
            <Button variant="contained" color="primary" size="small" onClick={handleUpdateDirectory}>
                Set Directory
            </Button>
            <List dense>
                {directories.map(dir => (
                    <ListItemButton key={dir} onClick={() => onSelect(dir)} dense divider>
                        <ListItemText primary={dir} />
                    </ListItemButton>
                ))}
            </List>
        </Container>
    );
}

export default DirectoryList;