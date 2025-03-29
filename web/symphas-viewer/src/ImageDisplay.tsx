import React, { useEffect, useState } from 'react';
import { Typography, Box, CircularProgress, ButtonGroup, Button, Slider } from '@mui/material';
import axios from 'axios';

interface ImageDisplayProps {
    directory: string;
    currentIndex: number;
}

interface SystemInfoProps {
    sizeX: number;
    sizeY: number;
    sizeZ: number;
    frames: number;
}
const ImageDisplay: React.FC<ImageDisplayProps> = ({ directory, currentIndex }) => {
    const [images, setImages] = useState<string[]>([]);
    const [loading, setLoading] = useState<boolean>(false);
    const [systemInfo, setSystemInfo] = useState<SystemInfoProps | null>(null);
    const [selectedAxis, setSelectedAxis] = useState<string>('Z');
    const [periodicIndex, setPeriodicIndex] = useState<number>(NaN);
    const [sliderValue, setSliderValue] = useState<number>(1);

    useEffect(() => {
        if (images.length > 0) {
            setPeriodicIndex(currentIndex % images.length);
        }
    }, [currentIndex, images]);

    useEffect(() => {
        setLoading(true);
        axios.get(`http://127.0.0.1:3030/get_system_size/${directory}/0`)
            .then(response => {
                const { len_x, len_y, len_z } = response.data;
                setSystemInfo({ sizeX: len_x, sizeY: len_y, sizeZ: len_z, frames: 0 });
            })
            .catch(error => {
                console.error('Error getting system size:', error);
            });
    }, [directory]);

    useEffect(() => {
        if (systemInfo) {
            setLoading(true);
            if (systemInfo.sizeZ > 1) {
                axios.get(`http://127.0.0.1:3030/get_slice/${directory}/0/${selectedAxis}/${sliderValue - 1}`)
                    .then(response => {
                        setImages(response.data);
                        setLoading(false);
                    })
                    .catch(error => {
                        console.error('Error fetching images:', error);
                        setImages([]);
                        setLoading(false);
                    });
            } else {
                axios.get(`http://127.0.0.1:3030/get_image/${directory}/0`)
                    .then(response => {
                        setImages(response.data);
                        setLoading(false);
                    })
                    .catch(error => {
                        console.error('Error fetching images:', error);
                        setImages([]);
                        setLoading(false);
                    });
            }
        }
    }, [systemInfo, selectedAxis, sliderValue]);

    const handleAxisChange = (axis: string) => {
        setSelectedAxis(axis);
        setSliderValue(1); // Reset slider value when axis changes
    };

    const getMaxSliderValue = () => {
        if (!systemInfo) return 1;
        switch (selectedAxis) {
            case 'X':
                return systemInfo.sizeX;
            case 'Y':
                return systemInfo.sizeY;
            case 'Z':
                return systemInfo.sizeZ;
            default:
                return 1;
        }
    };

    const handleSliderChange = (event: Event, newValue: number | number[]) => {
        setSliderValue(newValue as number);
    };

    return (
        <Box>
            {systemInfo && systemInfo.sizeZ > 1 && (
                <Box>

                    <ButtonGroup variant="contained" aria-label="outlined primary button group">
                        <Button onClick={() => handleAxisChange('X')} disabled={selectedAxis === 'X'}>X</Button>
                        <Button onClick={() => handleAxisChange('Y')} disabled={selectedAxis === 'Y'}>Y</Button>
                        <Button onClick={() => handleAxisChange('Z')} disabled={selectedAxis === 'Z'}>Z</Button>
                    </ButtonGroup>
                    <Slider
                        value={sliderValue}
                        min={1}
                        max={getMaxSliderValue()}
                        onChange={handleSliderChange}
                        aria-labelledby="axis-slider"
                        valueLabelDisplay="auto"
                    />
                </Box>
            )}
            {loading ? (
                <Box display="flex" justifyContent="center" alignItems="center" height="100%">
                    <CircularProgress />
                </Box>
            ) : (
                images.length > 0 ? (
                    <Box
                        component="img"
                        src={`data:image/png;base64,${images[periodicIndex]}`}
                        alt={`Image ${periodicIndex + 1} for ${directory}`}
                        sx={{ width: '100%', height: 'auto', border: 1 }}
                    />

                )
                    : (<Typography variant="body1">No images found for {directory}</Typography>)
            )}
        </Box >
    );
};

export default ImageDisplay;