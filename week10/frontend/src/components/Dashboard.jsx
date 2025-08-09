import React, { useEffect, useState } from 'react';
import axios from 'axios';
import EventList from './EventList';
import PriceChart from './PriceChart';
import DatePicker from 'react-datepicker';
import 'react-datepicker/dist/react-datepicker.css';
import '../dashboard.css'

const Dashboard = () => {
    const [data, setData] = useState([]);
    const [originalData, setOriginalData] = useState([]);
    const [events, setEvents] = useState([]);
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date());

    useEffect(() => {
        const fetchData = async () => {
            try {
                const result = await axios('http://127.0.0.1:5000/api/brent_data');
                if (Array.isArray(result.data)) {
                    setData(result.data);
                    setOriginalData(result.data);
                    const events = result.data.map(item => {
                        if (item.Event && item.Event.Description) {
                            return item.Event.Description; // Adjust based on your actual structure
                        }
                        return 'Unknown Event'; // Fallback for undefined events
                    });
                    console.log('Extracted Events:', events);
                    setEvents(events);
                } else {
                    console.error('Data is not an array:', result.data);
                }
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };
        fetchData();
    }, []);

    const filterData = () => {
        const filteredData = originalData.filter(item => {
            const itemDate = new Date(item.Date);
            return itemDate >= startDate && itemDate <= endDate;
        });
        
        const filteredEvents = events.filter(event => {
            const eventDate = new Date(event.Date);
            return eventDate >= startDate && eventDate <= endDate;
        });
    
        setData(filteredData);
        setEvents(filteredEvents);
    };

    return (
        <>
        <div className='dashboardHome'>
         <div>
         <h1>Brent Oil Prices Dashboard</h1>
        {/* Filtering Options */}
        <div className="filter-options">
            <h3>Filter by Date Range</h3>
            <DatePicker
                className="date-picker"
                selected={startDate}
                onChange={date => setStartDate(date)}
                selectsStart
                startDate={startDate}
                endDate={endDate}
            />
            <DatePicker
                className="date-picker"
                selected={endDate}
                onChange={date => setEndDate(date)}
                selectsEnd
                startDate={startDate}
                endDate={endDate}
                minDate={startDate}
            />
            <button className="filter-button" onClick={filterData}>Filter</button>
        </div>
    </div>
        <div>
            <PriceChart data={data.map(item => ({ date: item.Date, price: item.Price }))} />
            {/* <EventList events={events} /> */}
        </div>
        </div>
        </>
    );
};

export default Dashboard;