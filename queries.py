def get_intents(date: str, service_area: int = 1) -> str:
    return f"""
    with ios_radar_calls as (
        SELECT
            concat('i', userid) as userid,
            concat(
                '{date} ',
                cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)
            ) as ts,
            cast(latitude as double) as latitude,
            cast(longitude as double) as longitude
        FROM app_events.icma
        WHERE date = '{date}'
            and eventname = 'radar_call'
            and service_area_id = '{service_area}'
    ),
    ----------------------------------------------------------------------------------------------------------------------------
    android_radar_calls as (
        SELECT
            concat('a', userid) as userid,
            concat(
                '{date} ',
                cast(date_add('millisecond', cast(timestamp as bigint), TIME '00:00:00.000') as varchar)
            ) as ts,
            cast(latitude as double) as latitude,
            cast(longitude as double) as longitude
        FROM app_events.acma
        WHERE date = '{date}'
            and eventname = 'radar_call'
            and service_area_id = '{service_area}'
    ),
    ----------------------------------------------------------------------------------------------------------------------------
    data as (
        select *
        from ios_radar_calls as a
        union all
        select *
        from android_radar_calls as b
    )
    
    select *
    from data
    where 1=1
        and userid is not null
        and ts is not null
        and latitude is not null
        and longitude is not null
    """


def get_food_demand(start_day: str, end_day: str, city_id: int) -> str:
    return f"""
        select
            order_received_timestamp as ts,
            cast(order_id as varchar) as order_id,
            cast(booking_id as varchar) as booking_id,
            cast(customer_id as varchar) as customer_id,
            cast(captain_id as varchar) as captain_id,
            cct_id,
            cct,
            order_type,
            pickup_latitude,
            pickup_longitude,
            drop_off_latitude,
            drop_off_longitude,
            order_status,
            cancellation_type,
            cancelled_by,
            cancellation_reason
        from now_prod_dwh.orders
        where 1=1
            and day between date('{start_day}') and date('{end_day}')
            and city_id = {city_id}
            and delivery_type = 'careem'
    """
